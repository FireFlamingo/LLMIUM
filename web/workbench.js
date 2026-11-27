// Local navigation and explicit checkpoint-copy experiment controls.
window.workbenchUI = (()=>{
    const $=id=>document.getElementById(id);
    let selection=null, editingEnabled=false, polling=null, activeJob=null;
    const names={overview:'Overview',explorer:'Weight explorer',evidence:'Evidence & repair',agent:'Agent connection',project:'Project & demo'};
    function setView(view){
        for(const id of ['overview','agent-page','project-page','workspace']) $(id).hidden=true;
        $(view==='overview'?'overview':view==='agent'?'agent-page':view==='project'?'project-page':'workspace').hidden=false;
        document.querySelectorAll('[data-view]').forEach(button=>{button.classList.toggle('active',button.dataset.view===view);button.setAttribute('aria-current',button.dataset.view===view?'page':'false');});
        $('breadcrumb-view').textContent=names[view];
        $('workspace-heading').textContent=names[view];
        $('workspace-copy').textContent=view==='explorer'?'Select a checkpoint, inspect a tensor, and follow an individual parameter connection.':'Compare checkpoint findings, recorded generation and verified restoration.';
        $('report-controls').hidden=view==='explorer';$('checkpoint-controls').hidden=view!=='explorer';
    }
    function explore(fixture=false){
        setView('explorer');
        const options=[...$('atlas-model').options];
        const wanted=fixture?options.find(option=>option.textContent.includes('rust-demo/reference')):options.find(option=>option.textContent.includes('models/Qwen'));
        if(wanted)$('atlas-model').value=wanted.value;
        $('atlas-open').click();
    }
    function reportUpdated(data){
        const assessed=data.before||data;
        if(Array.isArray(assessed.tensors)){
            $('metric-parameters').textContent=assessed.parameters>=1e9?(assessed.parameters/1e9).toFixed(2)+'B':Number(assessed.parameters).toLocaleString();
            $('metric-tensors').textContent=assessed.tensors.length.toLocaleString();
            $('metric-changed').textContent=assessed.reference?assessed.tensors.reduce((sum,tensor)=>sum+(tensor.changed_elements||0),0).toLocaleString():'No baseline';
            $('metric-model').textContent=data.synthetic?'Synthetic fixture':assessed.checkpoint?.includes('qwen')?'Qwen2.5-1.5B-Instruct':assessed.checkpoint;
        }
        const recovered=data.recovery_verified??data.exact_forward_recovery;
        $('metric-recovery').textContent=recovered===true?'Verified':recovered===false?'Unverified':'Not measured';
        $('metric-recovery-note').textContent=data.report_type==='runtime_comparison'?'Checkpoint + recorded generation':'Selected tensor + measured probe';
    }
    function clearSelection(){selection=null;$('connection-editor').hidden=true;}
    function selectConnection(value){
        selection=value;$('connection-editor').hidden=false;
        $('edit-coordinate').textContent=value.tensor+' · row '+value.row+' · column '+value.column+' · flat index '+value.index;
        $('edit-original').textContent=String(value.value)+' · '+value.dtype+' · raw '+value.raw;
        $('edit-value').value=typeof value.value==='number'&&Number.isFinite(value.value)?String(value.value):'';
        $('edit-submit').disabled=!editingEnabled||activeJob!==null;
        $('edit-note').textContent=editingEnabled?'Creates a new checkpoint copy, compares the change, restores the tensor and reruns the selected demonstration.':'Editing is disabled. Start the Rust viewer with --allow-edits to run checkpoint-copy experiments.';
    }
    const phases={queued:'Queued',copying_and_editing:'Copying checkpoint and writing one parameter',assessing_change:'Scanning candidate against the baseline',restoring_and_verifying:'Restoring the tensor and verifying the checkpoint',measuring_outputs:'Measuring reference, candidate and repaired outputs',complete:'Experiment completed',failed:'Experiment failed'};
    function showJob(job){
        const panel=$('experiment-progress');panel.hidden=false;panel.replaceChildren();
        const title=document.createElement('strong');title.textContent=phases[job.phase]||job.phase;panel.append(title);
        const text=document.createElement('p');text.textContent=job.error||('Experiment '+job.id+' · '+job.mode+' · '+job.tensor+' ['+job.flat_index+']');panel.append(text);
        if(job.state==='complete'){
            const open=document.createElement('button');open.textContent='Open experiment evidence';open.onclick=()=>window.openSavedReport(job.report);panel.append(open);
            const inspect=document.createElement('button');inspect.textContent='Inspect edited checkpoint';inspect.onclick=()=>{setView('explorer');$('atlas-model').value=job.candidate_model;$('atlas-open').click();};panel.append(inspect);
        }
    }
    async function poll(){
        if(!activeJob)return;
        try{
            const response=await fetch('/api/jobs?id='+encodeURIComponent(activeJob));if(!response.ok)throw Error(await response.text());
            const job=await response.json();showJob(job);
            if(job.state!=='running'){
                clearInterval(polling);polling=null;activeJob=null;$('edit-submit').disabled=!editingEnabled||!selection;
                if(job.state==='complete')await window.atlasUI.refreshModels();
            }
        }catch(error){$('experiment-progress').textContent='Progress unavailable: '+error.message+'. The experiment may still be running; keep this page open.';}
    }
    async function submit(){
        if(!selection||activeJob)return;
        const raw=$('edit-value').value.trim(),value=Number(raw);
        if(!raw||!Number.isFinite(value)){$('edit-note').textContent='Enter a finite numeric value.';return;}
        $('edit-submit').disabled=true;
        try{
            const response=await fetch('/api/experiments',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:selection.model,tensor:selection.tensor,index:selection.index,value,mode:$('edit-mode').value,prompt:$('edit-prompt').value,steps:Number($('edit-steps').value)})});
            if(!response.ok)throw Error(await response.text());const job=await response.json();activeJob=job.id;showJob(job);polling=setInterval(poll,3000);
            $('edit-note').textContent='Experiment started. Large checkpoints require two new checkpoint copies and several minutes for verification and inference.';
        }catch(error){$('edit-note').textContent=error.message;$('edit-submit').disabled=!editingEnabled;}
    }
    document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{
        if(button.dataset.view==='explorer')explore();
        else if(button.dataset.view==='evidence'&&$('reports').value)window.openSavedReport($('reports').value);
        else setView(button.dataset.view);
    });
    document.querySelector('.brand').onclick=event=>{event.preventDefault();setView('overview');};
    document.querySelector('[data-action="explore"]').onclick=()=>explore();
    document.querySelector('[data-action="assess"]').onclick=()=>explore();
    document.querySelector('[data-action="fixture"]').onclick=()=>explore(true);
    $('edit-submit').onclick=submit;
    $('edit-mode').onchange=()=>{$('generation-settings').hidden=$('edit-mode').value!=='generation';};
    fetch('/api/jobs').then(response=>response.json()).then(data=>{
        editingEnabled=data.editing_enabled;$('engine-state').textContent='Rust engine connected';
        const running=data.jobs.find(job=>job.state==='running');if(running){activeJob=running.id;showJob(running);polling=setInterval(poll,3000);}
    }).catch(()=>{$('engine-state').textContent='Engine unavailable';});
    setView('overview');
    return {setView,reportUpdated,selectConnection,clearSelection};
})();
