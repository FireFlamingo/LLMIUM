// Read-only presentation of ordered activation observations recorded by Rust.
window.traceUI = (() => {
    const make = (tag, text, className) => {
        const element = document.createElement(tag);
        element.textContent = text;
        if (className) element.className = className;
        return element;
    };

    function attach(card, stage, evaluation) {
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
            const overlay = make('button', 'Show execution on checkpoint graph');
            controls.append(stepSelect, filter, faultLabel, overlay);
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
            function render() {
                const step = selectedStep();
                const records = step.activation_trace;
                const first = step.first_nonfinite_activation;
                summary.textContent = first ? 'First observed nonfinite output: ' + first.module + ' · ' + first.operation +
                    ' · ' + first.nonfinite + '/' + first.elements + ' elements.' :
                    'All ' + records.length + ' observed outputs were finite.';
                summary.className = first ? 'high' : 'good';
                const query = filter.value.toLowerCase();
                const filtered = records.filter(record => (!faults.checked || record.nonfinite > 0) &&
                    (record.module + ' ' + record.operation).toLowerCase().includes(query));
                const pages = Math.max(1, Math.ceil(filtered.length / 64));
                page = Math.min(page, pages - 1);
                previous.disabled = page === 0;next.disabled = page + 1 >= pages;
                pageStatus.textContent = filtered.length + ' observations · page ' + (page + 1) + '/' + pages;
                table.replaceChildren();
                const header = make('tr', '');
                for (const name of ['Order', 'Module / operation', 'Shape', 'Nonfinite']) header.append(make('th', name));
                table.append(header);
                for (const record of filtered.slice(page * 64, (page + 1) * 64)) {
                    const row = make('tr', '', record.nonfinite ? 'high' : '');
                    row.tabIndex = 0;
                    row.setAttribute('aria-label', 'Observation ' + record.sequence + ' ' + record.module + ' ' + record.operation);
                    for (const value of [record.sequence, record.module + ' · ' + record.operation, record.shape.join(' × '), record.nonfinite + '/' + record.elements]) row.append(make('td', value));
                    row.onclick = () => {detail.textContent = JSON.stringify(record, null, 2);};
                    row.onkeydown = event => {if(event.key === 'Enter')row.click();};
                    table.append(row);
                }
            }
            const reset = () => {page = 0;render();};
            stepSelect.onchange = reset;filter.oninput = reset;faults.onchange = reset;
            previous.onclick = () => {--page;render();};next.onclick = () => {++page;render();};
            overlay.onclick = () => window.atlasUI.openExecution(stage, selectedStep());
            render();
        };
    }
    return {attach};
})();
