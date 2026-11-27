// Read-only presentation of ordered activation observations recorded by Rust.
window.traceUI = (() => {
    const make = (tag, text, className) => {
        const element = document.createElement(tag);
        element.textContent = text;
        if (className) element.className = className;
        return element;
    };

    function compareRecord(record, reference) {
        const aligned = reference && typeof record.f32_sha256 === 'string' && typeof reference.f32_sha256 === 'string' &&
            record.module === reference.module && record.operation === reference.operation &&
            JSON.stringify(record.shape) === JSON.stringify(reference.shape);
        return {aligned:Boolean(aligned),changed:aligned ? record.f32_sha256 !== reference.f32_sha256 : null,
            reference_f32_sha256:aligned ? reference.f32_sha256 : null};
    }

    function attach(card, stage, evaluation, baseline) {
        if (!evaluation.steps.some(step => Array.isArray(step.activation_trace))) return;
        const panel = make('details', '');
        panel.append(make('summary', 'Recorded activation trace'));
        card.append(panel);
        let initialized = false;
        panel.ontoggle = () => {
            if (!panel.open || initialized) return;
            initialized = true;
            const controls = make('div', '', 'toolbar');
            const stepSelect = make('select', '');
            stepSelect.setAttribute('aria-label', 'Activation trace step');
            for (const [index, step] of evaluation.steps.entries()) {
                if (!Array.isArray(step.activation_trace)) continue;
                const option = make('option', 'Generation step ' + step.step);
                option.value = index;
                stepSelect.append(option);
            }
            const filter = make('input', '');
            filter.placeholder = 'Filter modules or operations';
            filter.setAttribute('aria-label', 'Filter activation trace');
            const faults = make('input', '');faults.type = 'checkbox';
            const faultLabel = make('label', '');faultLabel.append(faults, ' Only nonfinite outputs');
            const changes = make('input','');changes.type = 'checkbox';
            const changeLabel = make('label','');changeLabel.append(changes,' Only changed outputs');changeLabel.hidden = !baseline;
            const overlay = make('button', 'Show execution on checkpoint graph');
            controls.append(stepSelect, filter, faultLabel, changeLabel, overlay);
            const summary = make('p', '');
            const pageStatus = make('p', '', 'atlas-note');
            const previous = make('button', 'Previous trace page');
            const next = make('button', 'Next trace page');
            const paging = make('div', '', 'toolbar');paging.append(previous, next, pageStatus);
            const scroll = make('div', '', 'atlas-scroll');
            const table = make('table', '');table.setAttribute('aria-label', 'Recorded activation observations');scroll.append(table);
            const detail = make('pre', 'Select an observation for coordinates, hashes and finite-value statistics.');
            panel.append(controls, summary, paging, scroll, detail, make('p', stage.trace_coverage || 'Recorded intermediate outputs for this forward pass.'));
            let page = 0;
            const selectedStep = () => evaluation.steps[Number(stepSelect.value)];
            const referenceStep = () => baseline?.steps[Number(stepSelect.value)];
            const sameContext = () => Boolean(baseline && JSON.stringify(evaluation.input_token_ids) === JSON.stringify(baseline.input_token_ids) &&
                JSON.stringify(evaluation.generated_token_ids.slice(0,Number(stepSelect.value))) === JSON.stringify(baseline.generated_token_ids.slice(0,Number(stepSelect.value))));
            function render() {
                detail.textContent = 'Select an observation for coordinates, hashes and finite-value statistics.';
                const step = selectedStep();
                const records = step.activation_trace.map((record,index)=>({...record,reference_comparison:compareRecord(record,referenceStep()?.activation_trace?.[index])}));
                const first = step.first_nonfinite_activation;
                summary.textContent = first ? 'First observed nonfinite output: ' + first.module + ' · ' + first.operation +
                    ' · ' + first.nonfinite + '/' + first.elements + ' elements.' :
                    'All ' + records.length + ' observed outputs were finite.';
                summary.className = first ? 'high' : 'good';
                if (baseline) {
                    const changed = records.find(record=>record.reference_comparison.changed);
                    summary.textContent += changed ? ' First changed output: '+changed.module+' · '+changed.operation+'.' : ' No changed aligned outputs.';
                    summary.textContent += sameContext() ? ' Input context matches reference.' : ' Input context differs or reference is unavailable; differences alone do not localize the parameter change.';
                }
                const query = filter.value.toLowerCase();
                const filtered = records.filter(record => (!faults.checked || record.nonfinite > 0) &&
                    (!changes.checked || record.reference_comparison.changed === true) &&
                    (record.module + ' ' + record.operation).toLowerCase().includes(query));
                const pages = Math.max(1, Math.ceil(filtered.length / 64));
                page = Math.min(page, pages - 1);
                previous.disabled = page === 0;next.disabled = page + 1 >= pages;
                pageStatus.textContent = filtered.length + ' observations · page ' + (page + 1) + '/' + pages;
                table.replaceChildren();
                const header = make('tr', '');
                for (const name of ['Order', 'Module / operation', 'Shape', 'Nonfinite',...(baseline ? ['Reference'] : [])]) header.append(make('th', name));
                table.append(header);
                for (const record of filtered.slice(page * 64, (page + 1) * 64)) {
                    const row = make('tr', '', record.nonfinite ? 'high' : record.reference_comparison.changed ? 'changed-output' : '');
                    row.tabIndex = 0;
                    row.setAttribute('aria-label', 'Observation ' + record.sequence + ' ' + record.module + ' ' + record.operation);
                    for (const value of [record.sequence, record.module + ' · ' + record.operation, record.shape.join(' × '), record.nonfinite + '/' + record.elements]) row.append(make('td', value));
                    if (baseline) row.append(make('td',record.reference_comparison.aligned ? (record.reference_comparison.changed ? 'Changed' : 'Exact') : 'Unaligned'));
                    row.onclick = () => {detail.textContent = JSON.stringify(record, null, 2);};
                    row.onkeydown = event => {if(event.key === 'Enter')row.click();};
                    table.append(row);
                }
            }
            const reset = () => {page = 0;render();};
            stepSelect.onchange = reset;filter.oninput = reset;faults.onchange = reset;changes.onchange = reset;
            previous.onclick = () => {--page;render();};next.onclick = () => {++page;render();};
            overlay.onclick = () => window.atlasUI.openExecution(stage, selectedStep(), referenceStep(), sameContext());
            render();
        };
    }
    return {attach,compareRecord};
})();
