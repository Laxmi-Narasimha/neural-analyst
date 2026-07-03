'use client';

import { useMemo, useState } from 'react';
import styles from './TransformBuilder.module.css';

function uniqStrings(items) {
    const out = [];
    const seen = new Set();
    for (const raw of items || []) {
        const v = String(raw || '').trim();
        if (!v) continue;
        const key = v.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(v);
    }
    return out;
}

function splitCols(text) {
    return uniqStrings(String(text || '')
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean));
}

function mappingToText(mapping) {
    if (!mapping || typeof mapping !== 'object') return '';
    return Object.entries(mapping)
        .map(([k, v]) => `${String(k)}:${String(v)}`)
        .join('\n');
}

function parseMapping(text) {
    const mapping = {};
    const lines = String(text || '').split('\n');
    for (const line of lines) {
        const raw = String(line || '').trim();
        if (!raw) continue;
        const idx = raw.indexOf(':');
        if (idx <= 0) continue;
        const k = raw.slice(0, idx).trim();
        const v = raw.slice(idx + 1).trim();
        if (!k || !v) continue;
        mapping[k] = v;
    }
    return mapping;
}

function filterConditionsToText(conditions) {
    if (!Array.isArray(conditions)) return '';
    return conditions
        .map((c) => {
            const col = String(c?.column || '').trim();
            const op = String(c?.op || c?.operator || '').trim();
            const value = c?.value;
            let valueText = '';
            if (Array.isArray(value)) valueText = value.join(',');
            else if (value == null) valueText = '';
            else valueText = String(value);
            return [col, op, valueText].join('|');
        })
        .filter((line) => line.replace(/\|/g, '').trim())
        .join('\n');
}

function parseFilterConditions(text) {
    const out = [];
    for (const rawLine of String(text || '').split('\n')) {
        const line = String(rawLine || '').trim();
        if (!line) continue;
        const parts = line.split('|').map((s) => String(s || '').trim());
        const column = parts[0] || '';
        const op = (parts[1] || 'eq').toLowerCase();
        const rawValue = parts.slice(2).join('|');
        let value = rawValue;
        if (op === 'in' || op === 'not_in' || op === 'between') {
            value = rawValue
                .split(',')
                .map((s) => String(s || '').trim())
                .filter(Boolean);
        }
        out.push({ column, op, value });
    }
    return out;
}

function defaultStep(op) {
    const o = String(op || '').trim();
    if (o === 'deduplicate') return { op: 'deduplicate', params: { keep: 'first' } };
    if (o === 'type_convert') return { op: 'type_convert', params: { column: '', to: 'float', errors: 'coerce' } };
    if (o === 'fill_missing') return { op: 'fill_missing', params: { columns: [], strategy: 'constant', value: '' } };
    if (o === 'drop_missing') return { op: 'drop_missing', params: { columns: [], how: 'any' } };
    if (o === 'string_normalize') return { op: 'string_normalize', params: { columns: [], trim: true, lowercase: false, uppercase: false } };
    if (o === 'drop_columns') return { op: 'drop_columns', params: { columns: [] } };
    if (o === 'rename_columns') return { op: 'rename_columns', params: { mapping: {} } };
    if (o === 'filter_rows') return { op: 'filter_rows', params: { conditions: [{ column: '', op: 'eq', value: '' }], combine: 'all', keep: true } };
    if (o === 'sort_rows') return { op: 'sort_rows', params: { columns: [], ascending: true, na_position: 'last' } };
    if (o === 'limit_rows') return { op: 'limit_rows', params: { n: 1000, from_end: false } };
    if (o === 'time_features') return { op: 'time_features', params: { column: '', features: ['year', 'month', 'day', 'day_of_week', 'is_weekend'], prefix: '' } };
    if (o === 'bin_numeric') return { op: 'bin_numeric', params: { column: '', bins: 10, strategy: 'quantile', output: 'labels' } };
    if (o === 'clip_outliers') return { op: 'clip_outliers', params: { columns: [], method: 'iqr', action: 'clip', iqr_multiplier: 1.5 } };
    if (o === 'encode_categorical') return { op: 'encode_categorical', params: { columns: [], strategy: 'label', drop_original: false, max_categories: 100 } };
    return { op: o || 'deduplicate', params: {} };
}

function validateStep(step) {
    const op = String(step?.op || '').trim();
    const p = step?.params && typeof step.params === 'object' ? step.params : {};

    if (!op) return 'step.op is required';

    if (op === 'type_convert') {
        if (!String(p.column || '').trim()) return 'type_convert: column is required';
        if (!String(p.to || '').trim()) return 'type_convert: to is required';
        const errors = String(p.errors || 'raise').trim().toLowerCase();
        if (!['raise', 'coerce'].includes(errors)) return 'type_convert: errors must be raise|coerce';
    }

    if (op === 'fill_missing') {
        const cols = Array.isArray(p.columns) ? p.columns : splitCols(p.columns);
        if (!cols.length) return 'fill_missing: columns is required';
        const strategy = String(p.strategy || 'constant').trim().toLowerCase();
        if (!['constant', 'mean', 'median', 'mode', 'ffill', 'bfill'].includes(strategy)) return 'fill_missing: invalid strategy';
        if (strategy === 'constant' && !('value' in p)) return 'fill_missing: value is required for constant strategy';
    }

    if (op === 'drop_missing') {
        const how = String(p.how || 'any').trim().toLowerCase();
        if (!['any', 'all'].includes(how)) return 'drop_missing: how must be any|all';
    }

    if (op === 'deduplicate') {
        const keep = String(p.keep || 'first').trim().toLowerCase();
        if (!['first', 'last'].includes(keep)) return 'deduplicate: keep must be first|last';
    }

    if (op === 'string_normalize') {
        const cols = Array.isArray(p.columns) ? p.columns : splitCols(p.columns);
        if (!cols.length) return 'string_normalize: columns is required';
        if (Boolean(p.lowercase) && Boolean(p.uppercase)) return 'string_normalize: lowercase and uppercase are mutually exclusive';
        if (p.regex_replace != null && !Array.isArray(p.regex_replace)) return 'string_normalize: regex_replace must be a list';
    }

    if (op === 'drop_columns') {
        const cols = Array.isArray(p.columns) ? p.columns : splitCols(p.columns);
        if (!cols.length) return 'drop_columns: columns is required';
    }

    if (op === 'rename_columns') {
        const mapping = p.mapping;
        if (!mapping || typeof mapping !== 'object' || !Object.keys(mapping).length) return 'rename_columns: mapping is required';
    }

    if (op === 'filter_rows') {
        const conditions = Array.isArray(p.conditions) ? p.conditions : [];
        if (!conditions.length) return 'filter_rows: conditions are required';
        for (const c of conditions) {
            if (!String(c?.column || '').trim()) return 'filter_rows: each condition needs a column';
            if (!String(c?.op || '').trim()) return 'filter_rows: each condition needs an operator';
        }
        const combine = String(p.combine || 'all').toLowerCase();
        if (!['all', 'any'].includes(combine)) return 'filter_rows: combine must be all|any';
    }

    if (op === 'sort_rows') {
        const cols = Array.isArray(p.columns) ? p.columns : splitCols(p.columns);
        if (!cols.length) return 'sort_rows: columns are required';
        const na = String(p.na_position || 'last').toLowerCase();
        if (!['first', 'last'].includes(na)) return 'sort_rows: na_position must be first|last';
    }

    if (op === 'limit_rows') {
        const n = Number(p.n ?? p.limit ?? 0);
        if (!Number.isFinite(n) || n <= 0) return 'limit_rows: n must be positive';
    }

    if (op === 'time_features') {
        if (!String(p.column || '').trim()) return 'time_features: column is required';
        const features = Array.isArray(p.features) ? p.features : splitCols(p.features);
        if (!features.length) return 'time_features: at least one feature is required';
    }

    if (op === 'bin_numeric') {
        if (!String(p.column || '').trim()) return 'bin_numeric: column is required';
        const bins = Number(p.bins || 0);
        if (!Number.isFinite(bins) || bins < 2) return 'bin_numeric: bins must be >= 2';
        const strategy = String(p.strategy || 'quantile').toLowerCase();
        if (!['quantile', 'uniform'].includes(strategy)) return 'bin_numeric: strategy must be quantile|uniform';
    }

    if (op === 'clip_outliers') {
        const cols = Array.isArray(p.columns) ? p.columns : splitCols(p.columns);
        if (!cols.length) return 'clip_outliers: columns are required';
        const method = String(p.method || 'iqr').toLowerCase();
        if (!['iqr', 'quantile'].includes(method)) return 'clip_outliers: method must be iqr|quantile';
        const action = String(p.action || 'clip').toLowerCase();
        if (!['clip', 'drop'].includes(action)) return 'clip_outliers: action must be clip|drop';
    }

    if (op === 'encode_categorical') {
        const cols = Array.isArray(p.columns) ? p.columns : splitCols(p.columns);
        if (!cols.length) return 'encode_categorical: columns are required';
        const strategy = String(p.strategy || 'label').toLowerCase();
        if (!['label', 'one_hot'].includes(strategy)) return 'encode_categorical: strategy must be label|one_hot';
        const maxCategories = Number(p.max_categories || 0);
        if (!Number.isFinite(maxCategories) || maxCategories < 2) return 'encode_categorical: max_categories must be >= 2';
    }

    return null;
}

export default function TransformBuilder({ columns, steps, onChange, disabled }) {
    const cols = useMemo(() => uniqStrings((columns || []).map((c) => c?.name ?? c)), [columns]);
    const [addOp, setAddOp] = useState('deduplicate');
    const [localErrors, setLocalErrors] = useState([]);
    const [regexTextByIdx, setRegexTextByIdx] = useState({});

    const setSteps = (next) => {
        const arr = Array.isArray(next) ? next : [];
        const errs = [];
        for (let i = 0; i < arr.length; i += 1) {
            const msg = validateStep(arr[i]);
            if (msg) errs.push(`Step ${i + 1}: ${msg}`);
        }
        setLocalErrors(errs);
        onChange && onChange(arr, errs);
    };

    const move = (idx, dir) => {
        const arr = Array.isArray(steps) ? steps.slice() : [];
        const j = idx + dir;
        if (idx < 0 || idx >= arr.length) return;
        if (j < 0 || j >= arr.length) return;
        const tmp = arr[idx];
        arr[idx] = arr[j];
        arr[j] = tmp;
        setSteps(arr);
    };

    const remove = (idx) => {
        const arr = Array.isArray(steps) ? steps.slice() : [];
        arr.splice(idx, 1);
        setSteps(arr.length ? arr : [defaultStep('deduplicate')]);
    };

    const update = (idx, patch) => {
        const arr = Array.isArray(steps) ? steps.slice() : [];
        const base = arr[idx] && typeof arr[idx] === 'object' ? arr[idx] : defaultStep('deduplicate');
        const next = { ...base, ...patch };
        if (patch?.params) next.params = { ...(base.params || {}), ...(patch.params || {}) };
        arr[idx] = next;
        setSteps(arr);
    };

    const setOp = (idx, op) => {
        const arr = Array.isArray(steps) ? steps.slice() : [];
        arr[idx] = defaultStep(op);
        setSteps(arr);
    };

    const add = () => {
        const arr = Array.isArray(steps) ? steps.slice() : [];
        arr.push(defaultStep(addOp));
        setSteps(arr);
    };

    const renderColumnsInput = (idx, key, label, placeholder) => {
        const p = steps?.[idx]?.params || {};
        const raw = Array.isArray(p[key]) ? p[key].join(', ') : String(p[key] || '');
        return (
            <div className={styles.field}>
                <div className={styles.label}>{label}</div>
                <input
                    className={styles.input}
                    value={raw}
                    disabled={disabled}
                    placeholder={placeholder || 'col_a, col_b'}
                    onChange={(e) => update(idx, { params: { [key]: splitCols(e.target.value) } })}
                    list="na-columns"
                />
            </div>
        );
    };

    return (
        <div className={styles.wrap}>
            <datalist id="na-columns">
                {cols.map((c) => (
                    <option key={c} value={c} />
                ))}
            </datalist>

            <div className={styles.toolbar}>
                <div className={styles.toolbarLeft}>
                    <span className={styles.pill}>Transform Builder</span>
                    <select className={styles.select} value={addOp} disabled={disabled} onChange={(e) => setAddOp(e.target.value)}>
                        <option value="deduplicate">Deduplicate</option>
                        <option value="fill_missing">Fill missing</option>
                        <option value="drop_missing">Drop missing rows</option>
                        <option value="filter_rows">Filter rows</option>
                        <option value="sort_rows">Sort rows</option>
                        <option value="limit_rows">Limit rows</option>
                        <option value="type_convert">Type convert</option>
                        <option value="time_features">Time features</option>
                        <option value="bin_numeric">Bin numeric</option>
                        <option value="clip_outliers">Clip outliers</option>
                        <option value="encode_categorical">Encode categorical</option>
                        <option value="string_normalize">String normalize</option>
                        <option value="drop_columns">Drop columns</option>
                        <option value="rename_columns">Rename columns</option>
                    </select>
                    <button className={styles.btn} type="button" onClick={add} disabled={disabled}>
                        Add step
                    </button>
                </div>
                <span className={styles.pill}>{Array.isArray(steps) ? steps.length : 0} step(s)</span>
            </div>

            {localErrors?.length ? (
                <div className={styles.error}>
                    {localErrors.slice(0, 6).join(' | ')}
                    {localErrors.length > 6 ? ' | …' : ''}
                </div>
            ) : null}

            <div className={styles.list}>
                {(steps || []).map((step, idx) => {
                    const op = String(step?.op || 'deduplicate');
                    const p = step?.params && typeof step.params === 'object' ? step.params : {};

                    return (
                        <div key={`${op}-${idx}`} className={styles.step}>
                            <div className={styles.stepHead}>
                                <div className={styles.stepTitle}>
                                    <span className={styles.indexBadge}>{idx + 1}</span>
                                    <select
                                        className={styles.select}
                                        value={op}
                                        disabled={disabled}
                                        onChange={(e) => setOp(idx, e.target.value)}
                                    >
                                        <option value="deduplicate">Deduplicate</option>
                                        <option value="fill_missing">Fill missing</option>
                                        <option value="drop_missing">Drop missing rows</option>
                                        <option value="filter_rows">Filter rows</option>
                                        <option value="sort_rows">Sort rows</option>
                                        <option value="limit_rows">Limit rows</option>
                                        <option value="type_convert">Type convert</option>
                                        <option value="time_features">Time features</option>
                                        <option value="bin_numeric">Bin numeric</option>
                                        <option value="clip_outliers">Clip outliers</option>
                                        <option value="encode_categorical">Encode categorical</option>
                                        <option value="string_normalize">String normalize</option>
                                        <option value="drop_columns">Drop columns</option>
                                        <option value="rename_columns">Rename columns</option>
                                    </select>
                                </div>

                                <div className={styles.stepActions}>
                                    <button className={styles.miniBtn} type="button" disabled={disabled || idx === 0} onClick={() => move(idx, -1)}>
                                        Up
                                    </button>
                                    <button
                                        className={styles.miniBtn}
                                        type="button"
                                        disabled={disabled || idx === (steps?.length || 1) - 1}
                                        onClick={() => move(idx, 1)}
                                    >
                                        Down
                                    </button>
                                    <button className={styles.miniBtn} type="button" disabled={disabled} onClick={() => remove(idx)}>
                                        Remove
                                    </button>
                                </div>
                            </div>

                            <div className={styles.grid}>
                                {op === 'type_convert' ? (
                                    <>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Column</div>
                                            <input
                                                className={styles.input}
                                                value={String(p.column || '')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { column: e.target.value } })}
                                                placeholder="column name"
                                                list="na-columns"
                                            />
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Target type</div>
                                            <select
                                                className={styles.select}
                                                value={String(p.to || 'float')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { to: e.target.value } })}
                                            >
                                                <option value="int">int</option>
                                                <option value="float">float</option>
                                                <option value="str">string</option>
                                                <option value="bool">bool</option>
                                                <option value="datetime">datetime</option>
                                                <option value="date">date</option>
                                                <option value="category">category</option>
                                            </select>
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Errors</div>
                                            <select
                                                className={styles.select}
                                                value={String(p.errors || 'raise')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { errors: e.target.value } })}
                                            >
                                                <option value="raise">raise</option>
                                                <option value="coerce">coerce</option>
                                            </select>
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Datetime format (optional)</div>
                                            <input
                                                className={styles.input}
                                                value={String(p.format || '')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { format: e.target.value || null } })}
                                                placeholder="e.g. %Y-%m-%d"
                                            />
                                        </div>
                                    </>
                                ) : null}

                                {op === 'time_features' ? (
                                    <>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Datetime column</div>
                                            <input
                                                className={styles.input}
                                                value={String(p.column || '')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { column: e.target.value } })}
                                                placeholder="column name"
                                                list="na-columns"
                                            />
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Features</div>
                                            <input
                                                className={styles.input}
                                                value={Array.isArray(p.features) ? p.features.join(', ') : String(p.features || '')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { features: splitCols(e.target.value) } })}
                                                placeholder="year, month, day, day_of_week, is_weekend"
                                            />
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Prefix (optional)</div>
                                            <input
                                                className={styles.input}
                                                value={String(p.prefix || '')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { prefix: e.target.value } })}
                                                placeholder="e.g. order_"
                                            />
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Datetime format (optional)</div>
                                            <input
                                                className={styles.input}
                                                value={String(p.format || '')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { format: e.target.value || null } })}
                                                placeholder="e.g. %Y-%m-%d"
                                            />
                                        </div>
                                    </>
                                ) : null}

                                {op === 'bin_numeric' ? (
                                    <>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Numeric column</div>
                                            <input
                                                className={styles.input}
                                                value={String(p.column || '')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { column: e.target.value } })}
                                                placeholder="column name"
                                                list="na-columns"
                                            />
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Bins</div>
                                            <input
                                                className={styles.input}
                                                type="number"
                                                min={2}
                                                max={100}
                                                step={1}
                                                value={Number(p.bins || 10)}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { bins: Number(e.target.value || 0) } })}
                                            />
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Strategy</div>
                                            <select
                                                className={styles.select}
                                                value={String(p.strategy || 'quantile')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { strategy: e.target.value } })}
                                            >
                                                <option value="quantile">quantile</option>
                                                <option value="uniform">uniform</option>
                                            </select>
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Output</div>
                                            <select
                                                className={styles.select}
                                                value={String(p.output || 'labels')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { output: e.target.value } })}
                                            >
                                                <option value="labels">labels</option>
                                                <option value="codes">codes</option>
                                            </select>
                                        </div>
                                    </>
                                ) : null}

                                {op === 'clip_outliers' ? (
                                    <>
                                        {renderColumnsInput(idx, 'columns', 'Numeric columns', 'col_a, col_b')}
                                        <div className={styles.field}>
                                            <div className={styles.label}>Method</div>
                                            <select
                                                className={styles.select}
                                                value={String(p.method || 'iqr')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { method: e.target.value } })}
                                            >
                                                <option value="iqr">iqr</option>
                                                <option value="quantile">quantile</option>
                                            </select>
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Action</div>
                                            <select
                                                className={styles.select}
                                                value={String(p.action || 'clip')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { action: e.target.value } })}
                                            >
                                                <option value="clip">clip values</option>
                                                <option value="drop">drop rows</option>
                                            </select>
                                        </div>
                                        {String(p.method || 'iqr') === 'iqr' ? (
                                            <div className={styles.field}>
                                                <div className={styles.label}>IQR multiplier</div>
                                                <input
                                                    className={styles.input}
                                                    type="number"
                                                    min={0.1}
                                                    step={0.1}
                                                    value={Number(p.iqr_multiplier ?? 1.5)}
                                                    disabled={disabled}
                                                    onChange={(e) => update(idx, { params: { iqr_multiplier: Number(e.target.value || 0) } })}
                                                />
                                            </div>
                                        ) : (
                                            <>
                                                <div className={styles.field}>
                                                    <div className={styles.label}>Lower quantile</div>
                                                    <input
                                                        className={styles.input}
                                                        type="number"
                                                        min={0}
                                                        max={1}
                                                        step={0.01}
                                                        value={Number(p.lower_quantile ?? 0.01)}
                                                        disabled={disabled}
                                                        onChange={(e) => update(idx, { params: { lower_quantile: Number(e.target.value || 0) } })}
                                                    />
                                                </div>
                                                <div className={styles.field}>
                                                    <div className={styles.label}>Upper quantile</div>
                                                    <input
                                                        className={styles.input}
                                                        type="number"
                                                        min={0}
                                                        max={1}
                                                        step={0.01}
                                                        value={Number(p.upper_quantile ?? 0.99)}
                                                        disabled={disabled}
                                                        onChange={(e) => update(idx, { params: { upper_quantile: Number(e.target.value || 0) } })}
                                                    />
                                                </div>
                                            </>
                                        )}
                                        {String(p.action || 'clip') === 'drop' ? (
                                            <div className={styles.field}>
                                                <div className={styles.label}>Combine (drop rows)</div>
                                                <select
                                                    className={styles.select}
                                                    value={String(p.combine || 'any')}
                                                    disabled={disabled}
                                                    onChange={(e) => update(idx, { params: { combine: e.target.value } })}
                                                >
                                                    <option value="any">any outlier</option>
                                                    <option value="all">all columns outlier</option>
                                                </select>
                                            </div>
                                        ) : null}
                                    </>
                                ) : null}

                                {op === 'encode_categorical' ? (
                                    <>
                                        {renderColumnsInput(idx, 'columns', 'Categorical columns', 'col_a, col_b')}
                                        <div className={styles.field}>
                                            <div className={styles.label}>Strategy</div>
                                            <select
                                                className={styles.select}
                                                value={String(p.strategy || 'label')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { strategy: e.target.value } })}
                                            >
                                                <option value="label">label encoding</option>
                                                <option value="one_hot">one hot</option>
                                            </select>
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Max categories</div>
                                            <input
                                                className={styles.input}
                                                type="number"
                                                min={2}
                                                step={1}
                                                value={Number(p.max_categories ?? 100)}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { max_categories: Number(e.target.value || 0) } })}
                                            />
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Drop original</div>
                                            <label className={styles.check}>
                                                <input
                                                    type="checkbox"
                                                    checked={Boolean(p.drop_original)}
                                                    disabled={disabled}
                                                    onChange={(e) => update(idx, { params: { drop_original: e.target.checked } })}
                                                />
                                                drop source column(s)
                                            </label>
                                        </div>
                                        {String(p.strategy || 'label') === 'one_hot' ? (
                                            <>
                                                <div className={styles.field}>
                                                    <div className={styles.label}>Drop first level</div>
                                                    <label className={styles.check}>
                                                        <input
                                                            type="checkbox"
                                                            checked={Boolean(p.drop_first)}
                                                            disabled={disabled}
                                                            onChange={(e) => update(idx, { params: { drop_first: e.target.checked } })}
                                                        />
                                                        avoid full multicollinearity
                                                    </label>
                                                </div>
                                                <div className={styles.field}>
                                                    <div className={styles.label}>Create NA indicator</div>
                                                    <label className={styles.check}>
                                                        <input
                                                            type="checkbox"
                                                            checked={Boolean(p.dummy_na)}
                                                            disabled={disabled}
                                                            onChange={(e) => update(idx, { params: { dummy_na: e.target.checked } })}
                                                        />
                                                        include null category
                                                    </label>
                                                </div>
                                            </>
                                        ) : (
                                            <div className={styles.field}>
                                                <div className={styles.label}>Encoded suffix</div>
                                                <input
                                                    className={styles.input}
                                                    value={String(p.encoded_suffix || '_encoded')}
                                                    disabled={disabled}
                                                    onChange={(e) => update(idx, { params: { encoded_suffix: e.target.value } })}
                                                />
                                            </div>
                                        )}
                                    </>
                                ) : null}

                                {op === 'fill_missing' ? (
                                    <>
                                        {renderColumnsInput(idx, 'columns', 'Columns', 'col_a, col_b')}
                                        <div className={styles.field}>
                                            <div className={styles.label}>Strategy</div>
                                            <select
                                                className={styles.select}
                                                value={String(p.strategy || 'constant')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { strategy: e.target.value } })}
                                            >
                                                <option value="constant">constant</option>
                                                <option value="mean">mean</option>
                                                <option value="median">median</option>
                                                <option value="mode">mode</option>
                                                <option value="ffill">ffill</option>
                                                <option value="bfill">bfill</option>
                                            </select>
                                        </div>
                                        {String(p.strategy || 'constant') === 'constant' ? (
                                            <div className={styles.field}>
                                                <div className={styles.label}>Value</div>
                                                <input
                                                    className={styles.input}
                                                    value={String(p.value ?? '')}
                                                    disabled={disabled}
                                                    onChange={(e) => update(idx, { params: { value: e.target.value } })}
                                                    placeholder="value to fill"
                                                />
                                            </div>
                                        ) : null}
                                    </>
                                ) : null}

                                {op === 'drop_missing' ? (
                                    <>
                                        {renderColumnsInput(idx, 'columns', 'Columns (optional)', 'leave blank for all columns')}
                                        <div className={styles.field}>
                                            <div className={styles.label}>How</div>
                                            <select
                                                className={styles.select}
                                                value={String(p.how || 'any')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { how: e.target.value } })}
                                            >
                                                <option value="any">any</option>
                                                <option value="all">all</option>
                                            </select>
                                        </div>
                                    </>
                                ) : null}

                                {op === 'filter_rows' ? (
                                    <>
                                        <div className={styles.field} style={{ gridColumn: '1 / -1' }}>
                                            <div className={styles.label}>Conditions (one per line: column|op|value)</div>
                                            <textarea
                                                className={styles.textarea}
                                                value={filterConditionsToText(Array.isArray(p.conditions) ? p.conditions : [])}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { conditions: parseFilterConditions(e.target.value) } })}
                                                placeholder={'status|eq|active\nrevenue|gt|1000\ncategory|in|A,B,C'}
                                            />
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Combine</div>
                                            <select
                                                className={styles.select}
                                                value={String(p.combine || 'all')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { combine: e.target.value } })}
                                            >
                                                <option value="all">all conditions</option>
                                                <option value="any">any condition</option>
                                            </select>
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Keep rows</div>
                                            <label className={styles.check}>
                                                <input
                                                    type="checkbox"
                                                    checked={Boolean(p.keep ?? true)}
                                                    disabled={disabled}
                                                    onChange={(e) => update(idx, { params: { keep: e.target.checked } })}
                                                />
                                                keep matches
                                            </label>
                                        </div>
                                    </>
                                ) : null}

                                {op === 'sort_rows' ? (
                                    <>
                                        {renderColumnsInput(idx, 'columns', 'Columns', 'col_a, col_b')}
                                        <div className={styles.field}>
                                            <div className={styles.label}>Ascending</div>
                                            <label className={styles.check}>
                                                <input
                                                    type="checkbox"
                                                    checked={Boolean(p.ascending ?? true)}
                                                    disabled={disabled}
                                                    onChange={(e) => update(idx, { params: { ascending: e.target.checked } })}
                                                />
                                                ascending order
                                            </label>
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Null position</div>
                                            <select
                                                className={styles.select}
                                                value={String(p.na_position || 'last')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { na_position: e.target.value } })}
                                            >
                                                <option value="last">last</option>
                                                <option value="first">first</option>
                                            </select>
                                        </div>
                                    </>
                                ) : null}

                                {op === 'limit_rows' ? (
                                    <>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Rows to keep</div>
                                            <input
                                                className={styles.input}
                                                type="number"
                                                min={1}
                                                step={1}
                                                value={Number(p.n ?? p.limit ?? 1000)}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { n: Number(e.target.value || 0) } })}
                                            />
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>From end</div>
                                            <label className={styles.check}>
                                                <input
                                                    type="checkbox"
                                                    checked={Boolean(p.from_end)}
                                                    disabled={disabled}
                                                    onChange={(e) => update(idx, { params: { from_end: e.target.checked } })}
                                                />
                                                keep last N rows
                                            </label>
                                        </div>
                                    </>
                                ) : null}

                                {op === 'deduplicate' ? (
                                    <>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Keep</div>
                                            <select
                                                className={styles.select}
                                                value={String(p.keep || 'first')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { keep: e.target.value } })}
                                            >
                                                <option value="first">first</option>
                                                <option value="last">last</option>
                                            </select>
                                        </div>
                                        {renderColumnsInput(idx, 'subset', 'Subset columns (optional)', 'leave blank for all columns')}
                                    </>
                                ) : null}

                                {op === 'string_normalize' ? (
                                    <>
                                        {renderColumnsInput(idx, 'columns', 'Columns', 'col_a, col_b')}
                                        <div className={styles.field}>
                                            <div className={styles.label}>Options</div>
                                            <div className={styles.checkboxRow}>
                                                <label className={styles.check}>
                                                    <input
                                                        type="checkbox"
                                                        checked={Boolean(p.trim ?? true)}
                                                        disabled={disabled}
                                                        onChange={(e) => update(idx, { params: { trim: e.target.checked } })}
                                                    />
                                                    trim
                                                </label>
                                                <label className={styles.check}>
                                                    <input
                                                        type="checkbox"
                                                        checked={Boolean(p.lowercase)}
                                                        disabled={disabled}
                                                        onChange={(e) => update(idx, { params: { lowercase: e.target.checked } })}
                                                    />
                                                    lowercase
                                                </label>
                                                <label className={styles.check}>
                                                    <input
                                                        type="checkbox"
                                                        checked={Boolean(p.uppercase)}
                                                        disabled={disabled}
                                                        onChange={(e) => update(idx, { params: { uppercase: e.target.checked } })}
                                                    />
                                                    uppercase
                                                </label>
                                            </div>
                                        </div>
                                        <div className={styles.field}>
                                            <div className={styles.label}>Strip chars (optional)</div>
                                            <input
                                                className={styles.input}
                                                value={String(p.strip_chars || '')}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { strip_chars: e.target.value || null } })}
                                                placeholder="characters to strip"
                                            />
                                        </div>
                                        <div className={styles.field} style={{ gridColumn: '1 / -1' }}>
                                            <div className={styles.label}>Regex replacements (optional JSON list)</div>
                                            <textarea
                                                className={styles.textarea}
                                                value={
                                                    regexTextByIdx[idx] != null
                                                        ? String(regexTextByIdx[idx])
                                                        : p.regex_replace
                                                          ? JSON.stringify(p.regex_replace, null, 2)
                                                          : '[]'
                                                }
                                                disabled={disabled}
                                                onChange={(e) => {
                                                    const txt = e.target.value;
                                                    setRegexTextByIdx((prev) => ({ ...prev, [idx]: txt }));
                                                    try {
                                                        const parsed = JSON.parse(txt || '[]');
                                                        if (!Array.isArray(parsed)) return;
                                                        update(idx, { params: { regex_replace: parsed } });
                                                    } catch {
                                                        // keep text; validation will surface.
                                                    }
                                                }}
                                            />
                                        </div>
                                    </>
                                ) : null}

                                {op === 'drop_columns' ? (
                                    <>
                                        {renderColumnsInput(idx, 'columns', 'Columns', 'col_a, col_b')}
                                    </>
                                ) : null}

                                {op === 'rename_columns' ? (
                                    <>
                                        <div className={styles.field} style={{ gridColumn: '1 / -1' }}>
                                            <div className={styles.label}>Mapping (one per line: old:new)</div>
                                            <textarea
                                                className={styles.textarea}
                                                value={mappingToText(p.mapping)}
                                                disabled={disabled}
                                                onChange={(e) => update(idx, { params: { mapping: parseMapping(e.target.value) } })}
                                                placeholder={'old_name:new_name\ncol_a:col_a_clean'}
                                            />
                                        </div>
                                    </>
                                ) : null}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
