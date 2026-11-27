// The atlas uses only explicitly registered checkpoints and bounded read-only tile APIs.
window.atlasUI = (() => {
    const byId = id => document.getElementById(id);
    const make = (tag, text, className) => {
        const node = document.createElement(tag);
        node.textContent = text;
        if (className) node.className = className;
        return node;
    };
    const svgNode = (tag, attrs, text) => {
        const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
        for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, String(value));
        if (text !== undefined) node.textContent = text;
        return node;
    };
    let models = [];
    let data;
    let modelId = null;
    let layer = null;
    let tile;
    let selectedConnection=null;
    let execution = null;
    let graphEpoch = 0;
    let tileEpoch = 0;
    let cancelReport = () => {};

    async function request(path) {
        const response = await fetch(path);
        if (!response.ok) throw Error(await response.text());
        return response.json();
    }

    function cancel() { ++graphEpoch; ++tileEpoch; }

    async function openSelected() {
        cancelReport();
        cancel();
        const epoch = graphEpoch;
        const id = byId('atlas-model').value;
        byId('status').textContent = 'Reading checkpoint headers…';
        try {
            const report = await request('/api/checkpoint?model=' + encodeURIComponent(id));
            if (epoch !== graphEpoch) return;
            show(report, id);
        } catch (error) {
            if (epoch === graphEpoch) byId('status').textContent = error.message;
        }
    }

    async function openExecution(stage, step, baselineStep, inputContextMatches) {
        const model = models.find(model => model.path === stage.model_canonical);
        if (!model) {
            byId('status').textContent = 'Register the recorded checkpoint with --checkpoint to overlay this execution: ' + (stage.model_canonical || stage.model);
            return;
        }
        cancelReport();cancel();
        const epoch = graphEpoch;
        try {
            const report = await request('/api/checkpoint?model=' + model.id);
            if (epoch !== graphEpoch) return;
            if (report.config_sha256 !== stage.config_sha256) throw Error('Recorded configuration hash differs from the registered checkpoint.');
            show(report, String(model.id));
            step = {...step,activation_trace:step.activation_trace.map((record,index)=>({...record,
                reference_comparison:window.traceUI.compareRecord(record,baselineStep?.activation_trace?.[index])}))};
            execution = {stage, step};
            byId('atlas-execution').hidden = false;
            byId('atlas-execution-summary').textContent = 'Recorded generation step ' + step.step + ' · ' + step.activation_trace.length +
                ' observed outputs. Green modules had finite recorded outputs; orange modules had nonfinite outputs; purple modules had finite outputs changed from reference.' +
                (baselineStep ? (inputContextMatches ? ' Input context matches reference.' : ' Input context differs; output differences alone do not localize a parameter change.') : ' No reference comparison for this step.') +
                ' The overlay matches the registered path and configuration hash. Weight files are not rehashed by the viewer; keep checkpoints immutable.';
            const firstChanged = step.activation_trace.find(record=>record.reference_comparison.changed);
            byId('atlas-activation').textContent = JSON.stringify({model:stage.model_canonical,files:stage.files,input_context_matches:inputContextMatches,
                first_nonfinite_activation:step.first_nonfinite_activation,first_changed_activation:firstChanged}, null, 2);
            const first = (step.first_nonfinite_activation || firstChanged)?.module.match(/\.layers\.(\d+)/);
            if (first) layer = Number(first[1]);
            renderEndpoints();renderLayers();renderTensors();
        } catch (error) {
            if (epoch === graphEpoch) byId('status').textContent = error.message;
        }
    }

    function show(report, id) {
        window.workbenchUI.setView('explorer');
        cancel();
        data = report;
        execution = null;
        byId('atlas-execution').hidden = true;
        modelId = id;
        if (id !== null) byId('atlas-model').value = id;
        if (!Array.isArray(data.tensors) || !Array.isArray(data.layers)) throw Error('Atlas inventory is missing');
        layer = data.layers.length ? data.layers[0].index : null;
        byId('assessment').hidden = true;
        byId('runtime').hidden = true;
        byId('phase').hidden = true;
        byId('atlas').hidden = false;
        byId('status').textContent = 'CHECKPOINT ATLAS · ' + data.checkpoint;
        byId('atlas-summary').textContent = data.parameters.toLocaleString() + ' stored parameters · ' +
            data.tensors.length + ' tensors · ' + data.layers.length + ' layer groups';
        byId('atlas-basis').textContent = data.topology ? data.topology.basis :
            'Tensor inventory only. ' + (data.topology_error || 'No architecture adapter is available for this checkpoint.');
        byId('atlas-limits').textContent = data.interpretation;
        byId('atlas-filter').value = '';
        renderEndpoints();
        renderLayers();
        renderTensors();
    }

    function observationClass(records) {
        return records.length ? (records.some(record=>record.nonfinite > 0) ? ' observed-fault' :
            records.some(record=>record.reference_comparison?.changed) ? ' observed-changed' : ' observed-finite') : '';
    }

    function renderEndpoints() {
        byId('atlas-endpoints').replaceChildren();
        if (data.topology) {
            for (const [label, name, module] of [['Token embedding',data.topology.input_embedding,'model.embed_tokens'],
                ['Final RMSNorm',data.topology.final_norm,'model.norm'],
                [data.topology.tied_output_head ? 'LM head · tied embedding' : 'LM head',data.topology.output_head,'lm_head']]) {
                const records = execution?.step.activation_trace.filter(record=>record.module===module) || [];
                const button = make('button', label, observationClass(records).trim());
                button.onclick = () => {
                    layer = null; byId('atlas-filter').value=''; renderLayers(); renderTensors(name);
                    if (records.length) byId('atlas-activation').textContent = JSON.stringify(records,null,2);
                };
                byId('atlas-endpoints').append(button);
            }
            byId('atlas-endpoints').append(make('p','Tokens → embedding → decoder blocks in layer order → final RMSNorm → LM head', 'atlas-note'));
        }
    }

    function renderLayers() {
        byId('atlas-layers').replaceChildren();
        for (const item of data.layers) {
            const prefix = 'model.layers.' + item.index;
            const fault = execution?.step.activation_trace.some(record => (record.module === prefix || record.module.startsWith(prefix + '.')) && record.nonfinite > 0);
            const button = make('button', 'Layer ' + item.index + ' · ' + item.tensors.length + ' tensors',
                'node' + (layer === item.index ? ' active' : '') + (fault ? ' warn' : ''));
            button.onclick = () => { layer = item.index; byId('atlas-filter').value = ''; renderLayers(); renderTensors(); };
            byId('atlas-layers').append(button);
        }
        const all = make('button', 'All tensors', layer === null ? 'node active' : 'node');
        all.onclick = () => { layer = null; renderLayers(); renderTensors(); };
        byId('atlas-layers').append(all);
        renderBlock();
    }

    function renderBlock() {
        const svg = byId('atlas-block');
        svg.replaceChildren();
        const block = data.topology?.layers.find(block => block.index === layer);
        byId('atlas-block-heading').textContent = block ? 'Decoder block ' + layer : 'Module dataflow';
        if (!block) {
            svg.setAttribute('viewBox','0 0 900 100');
            svg.append(svgNode('text',{x:25,y:50}, data.topology ? 'Select a layer to inspect its module dataflow.' : 'No validated architecture topology for this checkpoint.'));
            return;
        }
        svg.setAttribute('viewBox','0 0 900 1260');
        const defs = svgNode('defs',{});
        const marker = svgNode('marker',{id:'atlas-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'});
        marker.append(svgNode('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#6c9b99'}));
        defs.append(marker); svg.append(defs);
        const nodes = new Map(block.nodes.map(node => [node.id,node]));
        for (const edge of block.edges) {
            const from = nodes.get(edge.from), to = nodes.get(edge.to);
            if (!from || !to) continue;
            const side = edge.from === 'input' ? 22 : 878;
            const endpoint = edge.from === 'input' ? to.x-123 : to.x+123;
            const path = edge.residual ? `M ${from.x} ${from.y} L ${side} ${from.y} L ${side} ${to.y} L ${endpoint} ${to.y}` :
                `M ${from.x} ${from.y+28} L ${to.x} ${to.y-28}`;
            svg.append(svgNode('path',{d:path,fill:'none',stroke:'#557b7d','stroke-width':2,'stroke-dasharray':edge.residual?'7 5':'none','marker-end':'url(#atlas-arrow)'}));
        }
        for (const node of block.nodes) {
            const records = observedOutputs(node);
            const observedClass = observationClass(records);
            const group = svgNode('g',{class:'module'+observedClass+(byId('atlas-tensor').value === node.tensor ? ' selected' : ''),
                ...(node.tensor || records.length ? {role:'button',tabindex:0,'aria-label':'Inspect '+node.label} : {})});
            group.append(svgNode('rect',{x:node.x-123,y:node.y-28,width:246,height:56,rx:7}));
            group.append(svgNode('text',{x:node.x,y:node.y+5,'text-anchor':'middle'},node.label));
            if (node.tensor || records.length) {
                group.append(svgNode('title',{},(node.tensor || node.label) + (records.length ? ' · ' + records.map(record=>record.operation+': '+record.nonfinite+'/'+record.elements+' nonfinite').join('; ') : '')));
                const select = () => {
                    if (records.length) byId('atlas-activation').textContent = JSON.stringify(records, null, 2);
                    if (node.tensor) {byId('atlas-filter').value=''; renderTensors(node.tensor);}
                };
                group.onclick = select;
                group.onkeydown = event => { if (event.key==='Enter'||event.key===' ') { event.preventDefault();select(); } };
            }
            svg.append(group);
        }
    }

    function observedOutputs(node) {
        if (!execution) return [];
        const prefix = 'model.layers.' + layer;
        const module = node.tensor?.replace(/\.(?:weight|bias)$/, '');
        return execution.step.activation_trace.filter(record => {
            if (module) return record.module === module;
            if (node.id === 'input') return layer === 0 ? record.module === 'model.embed_tokens' :
                record.module === 'model.layers.'+(layer-1) && record.operation === 'block_output';
            if (node.id === 'attention') return record.module === prefix+'.self_attn';
            if (node.id === 'gated') return record.module === prefix+'.mlp' && record.operation === 'gated_product';
            if (node.id === 'add1') return record.module === prefix && record.operation === 'attention_residual';
            if (node.id === 'add2' || node.id === 'output') return record.module === prefix && record.operation === 'block_output';
            return false;
        });
    }

    function renderTensors(preferred) {
        const query = byId('atlas-filter').value.toLowerCase();
        const candidates = data.tensors.filter(tensor =>
            (layer === null || tensor.layer === layer) && tensor.name.toLowerCase().includes(query));
        byId('atlas-tensor').replaceChildren(...candidates.map(tensor => {
            const option = make('option',tensor.name); option.value = tensor.name; return option;
        }));
        const initial = candidates.find(tensor => tensor.name === preferred) ||
            candidates.find(tensor => tensor.name.endsWith('.self_attn.q_proj.weight')) || candidates[0];
        if (initial) byId('atlas-tensor').value = initial.name;
        selectTensor();
    }

    function selectedTensor() { return data?.tensors.find(tensor => tensor.name === byId('atlas-tensor').value); }
    function dimensions(tensor) {
        if (tensor.shape.length > 2) throw Error('Tile rendering supports scalar, vector and rank-two tensors.');
        return tensor.shape.length === 2 ? tensor.shape : [1,tensor.shape[0] ?? 1];
    }

    function selectTensor() {
        selectedConnection=null;window.workbenchUI.clearSelection();
        ++tileEpoch;
        tile = null;
        byId('atlas-row').value = 0; byId('atlas-column').value = 0;
        byId('atlas-weight-table').replaceChildren(); byId('atlas-connections').replaceChildren();
        byId('atlas-cell').textContent = 'Select a weight to inspect its raw representation.';
        const tensor = selectedTensor();
        byId('atlas-tensor-shape').textContent = tensor ? tensor.shape.join(' × ')+' · '+tensor.dtype+' · '+tensor.parameters.toLocaleString()+' parameters' : 'No matching tensors.';
        renderBlock();
        if (!tensor) { byId('atlas-tile-status').textContent = 'No matching tensors.'; return; }
        if (modelId === null) {
            byId('atlas-tile-status').textContent = 'Offline metadata report. Inspect an explicitly registered checkpoint to read actual weight tiles.';
            return;
        }
        readTile();
    }

    async function readTile() {
        const tensor = selectedTensor();
        if (!tensor || modelId === null) return;
        const epoch = ++tileEpoch;
        try {
            const [height,width] = dimensions(tensor);
            const row = Number(byId('atlas-row').value), column = Number(byId('atlas-column').value);
            if (![row,column].every(Number.isSafeInteger) || row<0 || column<0 || row>=height || column>=width) throw Error('Choose row and column coordinates inside this tensor.');
            const size = Number(byId('atlas-size').value);
            const args = new URLSearchParams({model:modelId,tensor:tensor.name,row,column,rows:Math.min(size,height-row),columns:Math.min(size,width-column)});
            if (byId('atlas-reference').value !== '') args.set('reference',byId('atlas-reference').value);
            byId('atlas-tile-status').textContent = 'Reading actual weight bytes…';
            const result = await request('/api/tensor-tile?'+args);
            if (epoch !== tileEpoch) return;
            tile = result;
            selectedConnection=null;window.workbenchUI.clearSelection();
            byId('atlas-tile-status').textContent = tile.rows*tile.columns+' parameters in this tile · '+tile.nonfinite+' nonfinite'+
                (tile.changed_cells === null ? ' · No reference comparison' : ' · '+tile.changed_cells+' changed from reference');
            byId('atlas-cell').textContent = JSON.stringify({tile_sha256:tile.tile_sha256,coverage:tile.coverage},null,2);
            renderWeights();
            renderConnections();
        } catch (error) {
            if (epoch === tileEpoch) {
                tile = null; byId('atlas-weight-table').replaceChildren();byId('atlas-connections').replaceChildren();
                byId('atlas-tile-status').textContent = error.message;
            }
        }
    }

    const finite = value => typeof value === 'number' && Number.isFinite(value);
    function maxWeight() { return Math.max(1e-30,...tile.values.flat().filter(finite).map(Math.abs)); }
    function selectCell(row,column) {
        const y=tile.row+row, x=tile.column+column;
        byId('atlas-cell').textContent = JSON.stringify({tensor:tile.tensor,row:y,column:x,
            flat_index:y*tile.matrix_shape[1]+x,value:tile.values[row][column],raw_little_endian_hex:tile.raw_le_hex[row][column],
            reference_value:tile.reference_values?.[row]?.[column],changed:tile.changed?.[row]?.[column]},null,2);
        selectedConnection=[row,column];
        window.workbenchUI.selectConnection({model:Number(modelId),tensor:tile.tensor,row:y,column:x,index:y*tile.matrix_shape[1]+x,value:tile.values[row][column],raw:tile.raw_le_hex[row][column],dtype:selectedTensor().dtype});
        byId('edge-hover').textContent='Selected row '+y+', column '+x+' · value '+tile.values[row][column];
        renderConnections(selectedConnection);
    }

    function renderWeights() {
        const table = byId('atlas-weight-table'); table.replaceChildren();
        const header = make('tr',''); header.append(make('th','row / col'));
        for (let x=0;x<tile.columns;x++) header.append(make('th',tile.column+x));
        table.append(header);
        const maximum = maxWeight();
        for (let y=0;y<tile.rows;y++) {
            const row = make('tr','');row.append(make('th',tile.row+y));
            for (let x=0;x<tile.columns;x++) {
                const value = tile.values[y][x], valid = finite(value), changed=tile.changed?.[y]?.[x];
                const button = make('button',valid ? Number(value.toPrecision(3)).toString() : String(value),
                    (!valid?'fault ':'')+(changed?'drift':''));
                button.setAttribute('aria-label',`Weight row ${tile.row+y} column ${tile.column+x} value ${value}`);
                button.title = 'Raw '+tile.raw_le_hex[y][x]+(changed?' · changed from reference':'');
                if (valid) {
                    const alpha = .1+.6*Math.abs(value)/maximum;
                    button.style.backgroundColor = value>=0?`rgba(46,140,120,${alpha})`:`rgba(168,65,98,${alpha})`;
                }
                button.onclick = () => selectCell(y,x);
                const cell = make('td','');cell.append(button);row.append(cell);
            }
            table.append(row);
        }
    }

    function renderConnections(selected) {
        const svg=byId('atlas-connections');svg.replaceChildren();
        if (!tile) return;
        const maximum=maxWeight(), height=Math.max(tile.rows,tile.columns)*27+80;
        svg.setAttribute('viewBox',`0 0 720 ${height}`);
        for (let y=0;y<tile.rows;y++) for (let x=0;x<tile.columns;x++) {
            const value=tile.values[y][x], changed=tile.changed?.[y]?.[x], valid=finite(value);
            const chosen=selected && selected[0]===y && selected[1]===x;
            const line=svgNode('line',{x1:135,y1:45+x*27,x2:585,y2:45+y*27,
                stroke:chosen?'#ffffff':(!valid||changed?'#ffad76':value>=0?'#70e1be':'#e591b3'),
                'stroke-width':chosen?3:(!valid||changed?2:1),opacity:chosen||!valid||changed?1:.08+.65*Math.abs(value)/maximum});
            line.append(svgNode('title',{},`Row ${tile.row+y}, column ${tile.column+x}: ${value}`));
            const dy=(y-x)*27, length=Math.hypot(450,dy),ox=-dy/length*6,oy=450/length*6;
            const hit=svgNode('polygon',{points:`${135+ox},${45+x*27+oy} ${585+ox},${45+y*27+oy} ${585-ox},${45+y*27-oy} ${135-ox},${45+x*27-oy}`,class:'connection-hit',tabindex:0,role:'button',
                'aria-label':`Select connection row ${tile.row+y} column ${tile.column+x} value ${value}`,'aria-pressed':Boolean(chosen)});
            const enter=()=>{line.setAttribute('stroke','#e4b965');line.setAttribute('opacity','1');line.setAttribute('stroke-width','3');byId('edge-hover').textContent=`Row ${tile.row+y} · column ${tile.column+x} · flat index ${(tile.row+y)*tile.matrix_shape[1]+tile.column+x} · value ${value}`;};
            const leave=()=>{line.setAttribute('stroke',chosen?'#ffffff':(!valid||changed?'#ffad76':value>=0?'#70e1be':'#e591b3'));line.setAttribute('opacity',chosen||!valid||changed?'1':String(.08+.65*Math.abs(value)/maximum));line.setAttribute('stroke-width',chosen?'3':(!valid||changed?'2':'1'));};
            hit.onpointerenter=enter;hit.onpointerleave=leave;hit.onfocus=enter;hit.onblur=leave;
            hit.onclick=()=>selectCell(y,x);hit.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectCell(y,x);}};
            svg.append(line,hit);
        }
        for (const [count,column,x,textX,anchor] of [[tile.columns,true,135,120,'end'],[tile.rows,false,585,600,'start']]) {
            for (let i=0;i<count;i++) {
                svg.append(svgNode('circle',{cx:x,cy:45+i*27,r:4,fill:'#b6d0d1'}));
                svg.append(svgNode('text',{x:textX,y:49+i*27,'text-anchor':anchor},(column?'Column ':'Row ')+(column?tile.column+i:tile.row+i)));
            }
        }
    }

    function next(axis) {
        if (!tile) return;
        const input = byId(axis==='row'?'atlas-row':'atlas-column');
        const total = tile.matrix_shape[axis==='row'?0:1];
        const size = axis==='row'?tile.rows:tile.columns;
        input.value = Number(input.value)+size>=total ? 0 : Number(input.value)+size;
        readTile();
    }

    async function initialize(cancelPendingReport) {
        cancelReport = cancelPendingReport;
        byId('atlas-open').onclick = openSelected;
        byId('atlas-tensor').onchange = selectTensor;
        byId('atlas-filter').oninput = () => { if(data)renderTensors(); };
        byId('atlas-read').onclick = readTile;
        byId('atlas-reference').onchange = readTile;
        byId('atlas-size').onchange = readTile;
        byId('atlas-next-row').onclick = () => next('row');
        byId('atlas-next-column').onclick = () => next('column');
        byId('atlas-jump').onclick = () => {
            try {
                const tensor=selectedTensor(); if(!tensor)return;
                const [height,width]=dimensions(tensor), index=Number(byId('atlas-flat').value);
                if (!Number.isSafeInteger(index)||index<0||index>=height*width) throw Error('Flat index is outside this tensor.');
                byId('atlas-row').value=Math.floor(index/width);byId('atlas-column').value=index%width;readTile();
            }catch(error){byId('atlas-tile-status').textContent=error.message;}
        };
        await refreshModels();
    }
    async function refreshModels(){
        try {
            const current=byId('atlas-model').value,reference=byId('atlas-reference').value;
            models=await request('/api/checkpoints');
            byId('atlas-model').replaceChildren(...models.map(model=>{
                const option=make('option',model.id+' · '+model.path);option.value=model.id;return option;
            }));
            const noReference=make('option','No reference');noReference.value='';
            byId('atlas-reference').replaceChildren(noReference,...models.map(model=>{
                const option=make('option',model.id+' · '+model.path);option.value=model.id;return option;
            }));
            if(models.some(model=>String(model.id)===current))byId('atlas-model').value=current;
            if(models.some(model=>String(model.id)===reference))byId('atlas-reference').value=reference;
            byId('atlas-open').disabled=models.length===0;
            byId('atlas-registration').textContent=models.length ? models.length+' registered checkpoints · edits create copies' :
                'Register models with llmium serve --checkpoint PATH to inspect actual weights.';
        }catch(error){byId('atlas-open').disabled=true;byId('atlas-registration').textContent=error.message;}
    }
    return {initialize,show,cancel,openExecution,refreshModels};
})();
