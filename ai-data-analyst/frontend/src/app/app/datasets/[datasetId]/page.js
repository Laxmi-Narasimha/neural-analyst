'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { useParams, useSearchParams } from 'next/navigation';
import { motion } from 'framer-motion';
import api from '@/lib/api';
import { IconArrowRight, IconDatabase, IconDownload, IconLoader, IconSparkles, IconX } from '@/components/icons';
import TransformBuilder from '@/components/datasets/TransformBuilder';
import styles from './page.module.css';

function safeJson(value) {
    try {
        return JSON.stringify(value, null, 2);
    } catch {
        return String(value);
    }
}

function formatBytes(bytes) {
    const b = Number(bytes || 0);
    if (!Number.isFinite(b) || b <= 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let n = b;
    let u = 0;
    while (n >= 1024 && u < units.length - 1) {
        n /= 1024;
        u += 1;
    }
    return `${n.toFixed(u === 0 ? 0 : 1).replace('.0', '')} ${units[u]}`;
}

function shortHash(hash) {
    const h = String(hash || '');
    if (!h) return '';
    return h.length <= 12 ? h : `${h.slice(0, 12)}…`;
}

export default function DatasetDetailPage() {
    const params = useParams();
    const datasetId = params?.datasetId;
    const searchParams = useSearchParams();
    const prefillConsumedRef = useRef(false);

    const [dataset, setDataset] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [qualitySession, setQualitySession] = useState(null);
    const [versions, setVersions] = useState([]);
    const [versionsError, setVersionsError] = useState(null);
    const [versionsLoading, setVersionsLoading] = useState(false);

    const [preview, setPreview] = useState(null);
    const [previewLoading, setPreviewLoading] = useState(false);
    const [previewError, setPreviewError] = useState(null);

    const [colQuery, setColQuery] = useState('');

    const [sqlText, setSqlText] = useState('select * from dataset limit 25');
    const [sqlMaxRows, setSqlMaxRows] = useState('1000');
    const [sqlResult, setSqlResult] = useState(null);
    const [sqlBusy, setSqlBusy] = useState(false);
    const [sqlError, setSqlError] = useState(null);
    const [sqlRowsOffset, setSqlRowsOffset] = useState(0);
    const [sqlRowsLimit, setSqlRowsLimit] = useState(25);
    const [sqlRowsResult, setSqlRowsResult] = useState(null);
    const [sqlRowsBusy, setSqlRowsBusy] = useState(false);
    const [sqlRowsError, setSqlRowsError] = useState(null);

    const [transformMode, setTransformMode] = useState('builder'); // builder | json
    const [transformSteps, setTransformSteps] = useState([{ op: 'deduplicate', params: { keep: 'first' } }]);
    const [transformStepsText, setTransformStepsText] = useState(
        `[\n  { \"op\": \"deduplicate\", \"params\": { \"keep\": \"first\" } }\n]`
    );
    const [transformBuilderErrors, setTransformBuilderErrors] = useState([]);
    const [transformLabel, setTransformLabel] = useState('');
    const [transformSetCurrent, setTransformSetCurrent] = useState(true);
    const [transformPreview, setTransformPreview] = useState(null);
    const [transformJob, setTransformJob] = useState(null);
    const [transformJobState, setTransformJobState] = useState(null);
    const [transformBusy, setTransformBusy] = useState(false);
    const [transformSuggestBusy, setTransformSuggestBusy] = useState(false);
    const [transformSuggestionPack, setTransformSuggestionPack] = useState(null);
    const [transformError, setTransformError] = useState(null);

    const load = useCallback(async () => {
        if (!datasetId) return;
        try {
            setLoading(true);
            setError(null);
            const versionsRes = api.listDatasetVersions(datasetId, 1, 50).catch((e) => ({ items: [], __error: e }));
            const [d, latestQuality, v] = await Promise.all([
                api.getDataset(datasetId),
                api.getLatestQualitySession(datasetId).catch(() => null),
                versionsRes,
            ]);
            setDataset(d);
            setQualitySession(latestQuality);
            setVersions(Array.isArray(v?.items) ? v.items : []);
            setVersionsError(v?.__error ? v.__error?.message || 'Failed to load versions' : null);
        } catch (e) {
            setError(e?.message || 'Failed to load dataset');
        } finally {
            setLoading(false);
        }
    }, [datasetId]);

    useEffect(() => {
        load();
    }, [load]);

    useEffect(() => {
        if (prefillConsumedRef.current) return;
        const preset = String(searchParams?.get('transformPreset') || '').trim().toLowerCase();
        if (!preset) return;
        if (!dataset || String(dataset?.status || '').toLowerCase() !== 'ready') return;

        prefillConsumedRef.current = true;

        try {
            if (preset === 'fill_missing') {
                const raw = String(searchParams?.get('columns') || '').trim();
                const requested = raw
                    ? raw
                          .split(',')
                          .map((s) => String(s || '').trim())
                          .filter(Boolean)
                    : [];

                const info = Array.isArray(dataset?.columns) ? dataset.columns : [];
                const map = new Map(info.map((c) => [String(c?.name || ''), c]));

                const numeric = [];
                const nonNumeric = [];
                for (const c of requested) {
                    const meta = map.get(String(c));
                    const t = String(meta?.inferred_type || '').toLowerCase();
                    if (t === 'integer' || t === 'float') numeric.push(String(c));
                    else nonNumeric.push(String(c));
                }

                const steps = [];
                if (numeric.length) steps.push({ op: 'fill_missing', params: { columns: numeric, strategy: 'median' } });
                if (nonNumeric.length) steps.push({ op: 'fill_missing', params: { columns: nonNumeric, strategy: 'mode' } });
                if (!steps.length) return;

                setTransformMode('builder');
                setTransformSteps(steps);
                setTransformBuilderErrors([]);
                setTransformLabel(`fill_missing: ${requested.slice(0, 3).join(', ')}`);
                setTransformStepsText(JSON.stringify(steps, null, 2));
                setTransformError(null);
            } else if (preset === 'deduplicate') {
                const raw = String(searchParams?.get('subset') || '').trim();
                const subset = raw
                    ? raw
                          .split(',')
                          .map((s) => String(s || '').trim())
                          .filter(Boolean)
                    : [];

                const step = { op: 'deduplicate', params: { keep: 'first', ...(subset.length ? { subset } : {}) } };
                const steps = [step];

                setTransformMode('builder');
                setTransformSteps(steps);
                setTransformBuilderErrors([]);
                setTransformLabel(`deduplicate${subset.length ? `: ${subset.slice(0, 3).join(', ')}` : ''}`);
                setTransformStepsText(JSON.stringify(steps, null, 2));
                setTransformError(null);
            }
        } catch {
            // ignore prefill errors
        }
    }, [dataset, searchParams]);

    useEffect(() => {
        setSqlResult(null);
        setSqlError(null);
        setSqlRowsOffset(0);
        setSqlRowsResult(null);
        setSqlRowsError(null);
        setTransformSuggestionPack(null);
        setTransformJob(null);
        setTransformJobState(null);
    }, [datasetId]);

    const filteredColumns = useMemo(() => {
        const cols = dataset?.columns || [];
        const q = String(colQuery || '').trim().toLowerCase();
        if (!q) return cols;
        return cols.filter((c) => String(c?.name || '').toLowerCase().includes(q));
    }, [dataset, colQuery]);

    const refreshVersions = useCallback(async () => {
        if (!datasetId) return;
        try {
            setVersionsLoading(true);
            setVersionsError(null);
            const res = await api.listDatasetVersions(datasetId, 1, 50);
            setVersions(Array.isArray(res?.items) ? res.items : []);
        } catch (e) {
            setVersionsError(e?.message || 'Failed to load versions');
        } finally {
            setVersionsLoading(false);
        }
    }, [datasetId]);

    const activateVersion = async (versionId) => {
        if (!datasetId || !versionId) return;
        try {
            setVersionsLoading(true);
            setVersionsError(null);
            await api.activateDatasetVersion(datasetId, versionId);
            await load();
        } catch (e) {
            setVersionsError(e?.message || 'Failed to activate version');
        } finally {
            setVersionsLoading(false);
        }
    };

    const loadPreview = async () => {
        if (!datasetId) return;
        try {
            setPreviewLoading(true);
            setPreviewError(null);
            setPreview(null);

            const plan = [{ operator: 'preview_rows', params: { limit: 25 } }];
            const res = await api.runDataSpeaks(datasetId, plan, 25);

            const step = res?.steps?.[0];
            const artifact = step?.artifacts?.find((a) => a?.artifact_type === 'table') || step?.artifacts?.[0];
            const rows = artifact?.preview?.preview_rows || [];
            setPreview({ artifact, rows, summary: step?.summary });
        } catch (e) {
            setPreviewError(e?.message || 'Failed to load preview');
        } finally {
            setPreviewLoading(false);
        }
    };

    const loadSqlRows = useCallback(
        async (artifactId, offset, limit) => {
            const aid = String(artifactId || '').trim();
            if (!aid) return;
            const off = Math.max(0, Number(offset || 0));
            const lim = Math.max(1, Math.min(500, Number(limit || 25)));

            try {
                setSqlRowsBusy(true);
                setSqlRowsError(null);
                const rows = await api.getArtifactRows(aid, off, lim);
                setSqlRowsResult(rows || null);
                setSqlRowsOffset(off);
            } catch (e) {
                setSqlRowsError(e?.message || 'Failed to load artifact rows');
            } finally {
                setSqlRowsBusy(false);
            }
        },
        []
    );

    const runSql = async () => {
        if (!datasetId) return;
        const q = String(sqlText || '').trim();
        if (!q) {
            setSqlError('SQL query is required');
            return;
        }

        let maxRows = null;
        const maxRowsRaw = String(sqlMaxRows || '').trim();
        if (maxRowsRaw) {
            const n = Number(maxRowsRaw);
            if (!Number.isFinite(n) || n <= 0) {
                setSqlError('max rows must be a positive number');
                return;
            }
            maxRows = n;
        }

        try {
            setSqlBusy(true);
            setSqlError(null);
            setSqlResult(null);
            setSqlRowsResult(null);
            setSqlRowsError(null);
            setSqlRowsOffset(0);
            const res = await api.queryDatasetSql(datasetId, q, { maxRows });
            setSqlResult(res);
        } catch (e) {
            setSqlError(e?.message || 'Query failed');
        } finally {
            setSqlBusy(false);
        }
    };

    const suggestTransformPlan = async () => {
        if (!datasetId) return;
        try {
            setTransformSuggestBusy(true);
            setTransformError(null);
            const res = await api.suggestDatasetTransform(datasetId, { maxSteps: 10 });
            const rawSuggestions = Array.isArray(res?.suggestions) ? res.suggestions : [];
            const steps = rawSuggestions
                .map((s) => s?.step)
                .filter((s) => s && typeof s === 'object' && String(s.op || '').trim());

            setTransformSuggestionPack(res || null);
            if (!steps.length) {
                setTransformError('No transform suggestions generated for this dataset.');
                return;
            }

            setTransformMode('builder');
            setTransformSteps(steps);
            setTransformBuilderErrors([]);
            setTransformStepsText(JSON.stringify(steps, null, 2));
            if (!String(transformLabel || '').trim()) {
                setTransformLabel('auto_quality_plan');
            }
            setTransformPreview(null);
            setTransformJob(null);
        } catch (e) {
            setTransformError(e?.message || 'Failed to generate transform suggestions');
        } finally {
            setTransformSuggestBusy(false);
        }
    };

    const goSqlPage = async (nextOffset) => {
        const aid = String(sqlResult?.artifact?.id || '').trim();
        if (!aid) return;
        const off = Math.max(0, Number(nextOffset || 0));
        await loadSqlRows(aid, off, sqlRowsLimit);
    };

    useEffect(() => {
        const aid = String(sqlResult?.artifact?.id || '').trim();
        if (!aid) return;
        loadSqlRows(aid, 0, sqlRowsLimit);
    }, [sqlResult?.artifact?.id, sqlRowsLimit, loadSqlRows]);

    const getTransformSteps = () => {
        if (String(transformMode) === 'json') {
            try {
                const parsed = JSON.parse(transformStepsText || '[]');
                if (!Array.isArray(parsed)) throw new Error('Steps JSON must be an array');
                return parsed;
            } catch (e) {
                setTransformError(e?.message || 'Invalid steps JSON');
                return null;
            }
        }

        if (Array.isArray(transformBuilderErrors) && transformBuilderErrors.length) {
            setTransformError(transformBuilderErrors[0] || 'Invalid transformation steps');
            return null;
        }

        if (!Array.isArray(transformSteps) || !transformSteps.length) {
            setTransformError('Add at least one transformation step');
            return null;
        }

        return transformSteps;
    };

    const runTransformPreview = async () => {
        if (!datasetId) return;
        const steps = getTransformSteps();
        if (!steps) return;
        try {
            setTransformBusy(true);
            setTransformError(null);
            setTransformPreview(null);
            setTransformJob(null);
            const res = await api.previewDatasetTransform(datasetId, steps, 50_000, 25);
            setTransformPreview(res);
        } catch (e) {
            setTransformError(e?.message || 'Preview failed');
        } finally {
            setTransformBusy(false);
        }
    };

    const applyTransform = async () => {
        if (!datasetId) return;
        const steps = getTransformSteps();
        if (!steps) return;
        try {
            setTransformBusy(true);
            setTransformError(null);
            setTransformJob(null);
            setTransformJobState(null);
            const res = await api.applyDatasetTransform(datasetId, steps, {
                label: transformLabel || null,
                setAsCurrent: transformSetCurrent,
            });
            setTransformJob(res);
        } catch (e) {
            setTransformError(e?.message || 'Apply failed');
        } finally {
            setTransformBusy(false);
        }
    };

    useEffect(() => {
        const jobId = String(transformJob?.job_id || '').trim();
        if (!jobId) return undefined;

        let cancelled = false;
        let timer = null;
        const terminal = new Set(['completed', 'failed', 'cancelled']);

        const poll = async () => {
            try {
                const job = await api.getJob(jobId);
                if (cancelled) return;
                setTransformJobState(job || null);

                const statusText = String(job?.status || '').toLowerCase();
                if (terminal.has(statusText)) {
                    if (statusText === 'completed') {
                        setTransformError(null);
                        await load();
                    } else {
                        const msg = String(job?.error_message || job?.status_message || `Transform job ${statusText}`);
                        setTransformError(msg);
                    }
                    return;
                }

                timer = setTimeout(poll, 2500);
            } catch (e) {
                if (cancelled) return;
                setTransformError(e?.message || 'Failed to poll transform job status');
                timer = setTimeout(poll, 4000);
            }
        };

        poll();

        return () => {
            cancelled = true;
            if (timer) clearTimeout(timer);
        };
    }, [transformJob?.job_id, load]);

    const switchToJson = () => {
        try {
            setTransformStepsText(JSON.stringify(transformSteps || [], null, 2));
            setTransformMode('json');
            setTransformError(null);
        } catch {
            setTransformError('Failed to serialize steps');
        }
    };

    const switchToBuilder = () => {
        try {
            const parsed = JSON.parse(transformStepsText || '[]');
            if (!Array.isArray(parsed)) throw new Error('Steps JSON must be an array');
            setTransformSteps(parsed);
            setTransformBuilderErrors([]);
            setTransformMode('builder');
            setTransformError(null);
        } catch (e) {
            setTransformError(e?.message || 'Invalid steps JSON');
        }
    };

    const downloadArtifact = async (artifactId) => {
        try {
            const blob = await api.downloadArtifactData(artifactId);
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = String(artifactId);
            document.body.appendChild(a);
            a.click();
            a.remove();

            setTimeout(() => URL.revokeObjectURL(url), 10_000);
        } catch (e) {
            setError(e?.message || 'Download failed');
        }
    };

    const status = String(dataset?.status || '').toLowerCase();
    const ready = status === 'ready';
    const transformJobStatus = String(transformJobState?.status || '').toLowerCase();
    const transformJobInFlight = transformJobStatus === 'queued' || transformJobStatus === 'running';
    const piiColumns = (dataset?.columns || []).filter((c) => Boolean(c?.statistics?.is_potential_pii));
    const readiness = useMemo(() => {
        if (!qualitySession) return { label: 'Not run', status: 'none' };
        const st = String(qualitySession.status || '').toLowerCase();
        const level = qualitySession.readiness_level ? String(qualitySession.readiness_level) : '';
        const score = qualitySession.composite_score;
        let label = level || st || 'unknown';
        if (score != null && Number.isFinite(Number(score))) {
            label = `${label} (${Number(score).toFixed(2)})`;
        }
        return { label, status: st || 'unknown', sessionId: qualitySession.session_id };
    }, [qualitySession]);

    return (
        <div className={styles.page}>
            <div className={styles.header}>
                <Link href="/app/datasets" className={styles.back}>
                    <IconArrowRight size={18} className={styles.backIcon} />
                    Datasets
                </Link>

                <div className={styles.titleRow}>
                    <div className={styles.titleIcon}>
                        <IconDatabase size={18} />
                    </div>
                    <div className={styles.titleBlock}>
                        <h1 className={styles.title}>{dataset?.name || 'Dataset'}</h1>
                        <div className={styles.subtitle}>
                            <span className={`${styles.badge} ${styles[status]}`}>{status || 'unknown'}</span>
                            {dataset?.original_filename ? <span>{dataset.original_filename}</span> : null}
                        </div>
                    </div>
                </div>

                <div className={styles.actions}>
                    <Link
                        href={ready ? `/app/analysis/new?dataset=${datasetId}` : '/app/datasets'}
                        className={styles.primaryBtn}
                        aria-disabled={!ready}
                        title={ready ? 'Open chat analysis' : 'Dataset not ready yet'}
                    >
                        <IconSparkles size={16} />
                        Talk to your data
                    </Link>
                    <Link
                        href={`/app/data-speaks?dataset=${datasetId}`}
                        className={styles.secondaryBtn}
                        title="Run Data Speaks (EDA)"
                    >
                        <IconSparkles size={16} />
                        Data Speaks
                    </Link>
                    <Link href="/app/quality" className={styles.secondaryBtn} title="Run data adequacy / quality">
                        <IconSparkles size={16} />
                        Quality
                    </Link>
                </div>
            </div>

            {error && (
                <motion.div className={styles.errorBanner} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
                    <IconX size={18} />
                    {error}
                    <button className={styles.dismiss} onClick={() => setError(null)} type="button">
                        Dismiss
                    </button>
                </motion.div>
            )}

            {loading ? (
                <div className={styles.loading}>
                    <IconLoader size={28} className={styles.spinning} />
                    <p>Loading dataset…</p>
                </div>
            ) : dataset ? (
                <>
                    <div className={styles.metaGrid}>
                        <div className={styles.metaCard}>
                            <div className={styles.metaLabel}>Format</div>
                            <div className={styles.metaValue}>{String(dataset.file_format || '').toUpperCase()}</div>
                        </div>
                        <div className={styles.metaCard}>
                            <div className={styles.metaLabel}>Size</div>
                            <div className={styles.metaValue}>{formatBytes(dataset.file_size_bytes)}</div>
                        </div>
                        <div className={styles.metaCard}>
                            <div className={styles.metaLabel}>Rows</div>
                            <div className={styles.metaValue}>{(dataset.row_count || 0).toLocaleString()}</div>
                        </div>
                        <div className={styles.metaCard}>
                            <div className={styles.metaLabel}>Columns</div>
                            <div className={styles.metaValue}>{dataset.column_count || 0}</div>
                        </div>
                        <div className={styles.metaCard}>
                            <div className={styles.metaLabel}>Readiness</div>
                            <div className={styles.metaValue}>
                                {readiness.label}
                                {readiness.sessionId ? (
                                    <span className={styles.mutedInline}> session {String(readiness.sessionId)}</span>
                                ) : null}
                            </div>
                        </div>
                    </div>

                    {piiColumns.length ? (
                        <div className={styles.notice}>
                            <strong>PII warning:</strong> potential PII detected in {piiColumns.length} column(s):{' '}
                            {piiColumns.slice(0, 8).map((c) => c.name).join(', ')}
                            {piiColumns.length > 8 ? '…' : ''}
                        </div>
                    ) : null}

                    <div className={styles.section} id="sql">
                        <div className={styles.sectionHeader}>
                            <h2 className={styles.sectionTitle}>Preview</h2>
                            <div className={styles.sectionActions}>
                                <button
                                    className={styles.secondaryBtn}
                                    onClick={loadPreview}
                                    type="button"
                                    disabled={!ready || previewLoading}
                                    title={ready ? 'Load preview rows' : 'Dataset not ready'}
                                >
                                    {previewLoading ? <IconLoader size={16} className={styles.spinning} /> : <IconSparkles size={16} />}
                                    Load preview rows
                                </button>
                                {preview?.artifact?.artifact_id ? (
                                    <button
                                        className={styles.secondaryBtn}
                                        onClick={() => downloadArtifact(preview.artifact.artifact_id)}
                                        type="button"
                                        title="Download preview artifact data"
                                    >
                                        <IconDownload size={16} />
                                        Download
                                    </button>
                                ) : null}
                            </div>
                        </div>

                        {previewError ? (
                            <div className={styles.muted}>{previewError}</div>
                        ) : preview?.rows?.length ? (
                            <div className={styles.previewTableWrap}>
                                <table className={styles.table}>
                                    <thead>
                                        <tr>
                                            {Object.keys(preview.rows[0] || {}).map((k) => (
                                                <th key={k}>{k}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {preview.rows.map((row, idx) => (
                                            <tr key={idx}>
                                                {Object.keys(preview.rows[0] || {}).map((k) => (
                                                    <td key={k}>{String(row?.[k] ?? '')}</td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : (
                            <div className={styles.muted}>
                                {ready ? 'Click “Load preview rows” to fetch a safe preview.' : 'Dataset is not ready yet.'}
                            </div>
                        )}

                        {preview?.summary ? <pre className={styles.pre}>{safeJson(preview.summary)}</pre> : null}
                    </div>

                    <div className={styles.section} id="transform">
                        <div className={styles.sectionHeader}>
                            <h2 className={styles.sectionTitle}>SQL (read-only)</h2>
                            <div className={styles.sectionActions}>
                                <input
                                    className={styles.input}
                                    style={{ maxWidth: 160 }}
                                    value={sqlMaxRows}
                                    onChange={(e) => setSqlMaxRows(e.target.value)}
                                    disabled={!ready || sqlBusy}
                                    inputMode="numeric"
                                    placeholder="max rows"
                                    title="Maximum rows returned (server enforced)"
                                />
                                <input
                                    className={styles.input}
                                    style={{ maxWidth: 150 }}
                                    value={String(sqlRowsLimit)}
                                    onChange={(e) => {
                                        const n = Number(e.target.value);
                                        if (!Number.isFinite(n)) return;
                                        setSqlRowsLimit(Math.max(1, Math.min(500, Math.floor(n))));
                                    }}
                                    disabled={!ready || sqlBusy || sqlRowsBusy}
                                    inputMode="numeric"
                                    placeholder="rows/page"
                                    title="Rows per page for artifact table viewer"
                                />
                                <button
                                    className={styles.secondaryBtn}
                                    onClick={runSql}
                                    type="button"
                                    disabled={!ready || sqlBusy}
                                    title={ready ? 'Run SQL query' : 'Dataset not ready'}
                                >
                                    {sqlBusy ? <IconLoader size={16} className={styles.spinning} /> : <IconSparkles size={16} />}
                                    Run query
                                </button>
                                {sqlResult?.artifact?.id ? (
                                    <button
                                        className={styles.secondaryBtn}
                                        onClick={() => downloadArtifact(sqlResult.artifact.id)}
                                        type="button"
                                        title="Download query result artifact data"
                                    >
                                        <IconDownload size={16} />
                                        Download
                                    </button>
                                ) : null}
                                {sqlResult?.artifact?.id ? (
                                    <Link
                                        className={styles.secondaryBtn}
                                        href={`/app/artifacts/${encodeURIComponent(String(sqlResult.artifact.id))}`}
                                        title="Open artifact viewer"
                                    >
                                        Open
                                    </Link>
                                ) : null}
                            </div>
                        </div>

                        <div className={styles.muted}>
                            Query your uploaded dataset using DuckDB. Use <code>dataset</code> as the table name. File access and unsafe functions are blocked.
                        </div>

                        {sqlError ? <div className={styles.noticeError} style={{ marginTop: 12 }}>{sqlError}</div> : null}

                        <textarea
                            className={styles.textarea}
                            value={sqlText}
                            onChange={(e) => setSqlText(e.target.value)}
                            disabled={!ready || sqlBusy}
                            spellCheck={false}
                        />

                        {sqlResult ? (
                            <>
                                <div className={styles.muted} style={{ marginTop: 10 }}>
                                    <strong>Rows:</strong> {Number(sqlResult.row_count || 0).toLocaleString()} &nbsp;|&nbsp;{' '}
                                    <strong>Time:</strong> {Number(sqlResult.execution_time_ms || 0).toFixed(1)}ms
                                </div>
                                {sqlRowsError ? <div className={styles.noticeError} style={{ marginTop: 12 }}>{sqlRowsError}</div> : null}

                                {sqlRowsResult?.rows?.length ? (
                                    <>
                                        <div className={styles.sectionActions} style={{ marginTop: 12 }}>
                                            <button
                                                className={styles.secondaryBtn}
                                                type="button"
                                                onClick={() => goSqlPage(Math.max(0, sqlRowsOffset - sqlRowsLimit))}
                                                disabled={sqlRowsBusy || sqlRowsOffset <= 0}
                                            >
                                                Prev
                                            </button>
                                            <button
                                                className={styles.secondaryBtn}
                                                type="button"
                                                onClick={() => goSqlPage(sqlRowsOffset + sqlRowsLimit)}
                                                disabled={
                                                    sqlRowsBusy ||
                                                    (Number.isFinite(Number(sqlRowsResult.total_rows))
                                                        ? (sqlRowsOffset + sqlRowsLimit) >= Number(sqlRowsResult.total_rows)
                                                        : (sqlRowsResult.rows || []).length < sqlRowsLimit)
                                                }
                                            >
                                                Next
                                            </button>
                                            <span className={styles.mutedInline}>
                                                Showing {sqlRowsOffset + 1}
                                                {' - '}
                                                {sqlRowsOffset + (sqlRowsResult.rows || []).length}
                                                {Number.isFinite(Number(sqlRowsResult.total_rows))
                                                    ? ` of ${Number(sqlRowsResult.total_rows).toLocaleString()}`
                                                    : ''}
                                            </span>
                                            {sqlRowsBusy ? <IconLoader size={14} className={styles.spinning} /> : null}
                                        </div>

                                        <div className={styles.previewTableWrap} style={{ marginTop: 12 }}>
                                            <table className={styles.table}>
                                                <thead>
                                                    <tr>
                                                        {(sqlRowsResult.columns || []).map((k) => (
                                                            <th key={k}>{k}</th>
                                                        ))}
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {(sqlRowsResult.rows || []).map((row, idx) => (
                                                        <tr key={idx}>
                                                            {(sqlRowsResult.columns || []).map((k) => (
                                                                <td key={k}>{String(row?.[k] ?? '')}</td>
                                                            ))}
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </>
                                ) : sqlResult?.artifact?.preview?.preview_rows?.length ? (
                                    <div className={styles.previewTableWrap} style={{ marginTop: 12 }}>
                                        <table className={styles.table}>
                                            <thead>
                                                <tr>
                                                    {Object.keys(sqlResult.artifact.preview.preview_rows[0] || {}).map((k) => (
                                                        <th key={k}>{k}</th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {sqlResult.artifact.preview.preview_rows.map((row, idx) => (
                                                    <tr key={idx}>
                                                        {Object.keys(sqlResult.artifact.preview.preview_rows[0] || {}).map((k) => (
                                                            <td key={k}>{String(row?.[k] ?? '')}</td>
                                                        ))}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                ) : (
                                    <div className={styles.muted} style={{ marginTop: 12 }}>No rows returned.</div>
                                )}
                            </>
                        ) : null}
                    </div>

                    <div className={styles.section}>
                        <div className={styles.sectionHeader}>
                            <h2 className={styles.sectionTitle}>Versions</h2>
                            <div className={styles.sectionActions}>
                                <button
                                    className={styles.secondaryBtn}
                                    onClick={refreshVersions}
                                    type="button"
                                    disabled={versionsLoading}
                                    title="Refresh versions"
                                >
                                    {versionsLoading ? <IconLoader size={16} className={styles.spinning} /> : <IconSparkles size={16} />}
                                    Refresh
                                </button>
                            </div>
                        </div>

                        {versionsError ? <div className={styles.muted}>{versionsError}</div> : null}
                        {!versionsLoading && (!Array.isArray(versions) || versions.length === 0) ? (
                            <div className={styles.muted}>No versions indexed yet. Process the dataset to create the first version.</div>
                        ) : null}

                        <div className={styles.versionList}>
                            {(versions || []).map((v) => (
                                <div key={v.id} className={styles.versionCard}>
                                    <div className={styles.versionInfo}>
                                        <div className={styles.versionTitle}>{v.label || (v.transform_spec?.steps ? 'Transformed' : 'Version')}</div>
                                        <div className={styles.versionMeta}>
                                            <span className={styles.badgeMuted}>{shortHash(v.version_hash)}</span>
                                            <span>{String(v.file_format || '').toUpperCase()}</span>
                                            {v.row_count != null ? <span>{Number(v.row_count).toLocaleString()} rows</span> : null}
                                            {v.column_count != null ? <span>{Number(v.column_count)} cols</span> : null}
                                            {v.created_at ? <span>{new Date(v.created_at).toLocaleString()}</span> : null}
                                        </div>
                                    </div>
                                    <div className={styles.versionActions}>
                                        {v.is_active ? (
                                            <span className={styles.activePill}>Active</span>
                                        ) : (
                                            <button
                                                className={styles.secondaryBtn}
                                                onClick={() => activateVersion(v.id)}
                                                type="button"
                                                disabled={versionsLoading}
                                                title="Activate this version"
                                            >
                                                Activate
                                            </button>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className={styles.section}>
                        <div className={styles.sectionHeader}>
                            <h2 className={styles.sectionTitle}>Transform</h2>
                            <div className={styles.sectionActions}>
                                <button
                                    className={styles.secondaryBtn}
                                    onClick={suggestTransformPlan}
                                    type="button"
                                    disabled={!ready || transformBusy || transformSuggestBusy || transformJobInFlight}
                                    title={ready ? 'Generate a deterministic cleaning plan from dataset metadata' : 'Dataset not ready'}
                                >
                                    {transformSuggestBusy ? <IconLoader size={16} className={styles.spinning} /> : <IconSparkles size={16} />}
                                    Suggest plan
                                </button>
                                <button
                                    className={styles.secondaryBtn}
                                    onClick={runTransformPreview}
                                    type="button"
                                    disabled={!ready || transformBusy || transformSuggestBusy || transformJobInFlight}
                                    title={ready ? 'Preview transformation on a sample' : 'Dataset not ready'}
                                >
                                    {transformBusy ? <IconLoader size={16} className={styles.spinning} /> : <IconSparkles size={16} />}
                                    Preview
                                </button>
                                <button
                                    className={styles.primaryBtn}
                                    onClick={applyTransform}
                                    type="button"
                                    disabled={!ready || transformBusy || transformSuggestBusy || transformJobInFlight}
                                    title={ready ? 'Create a new version (async job)' : 'Dataset not ready'}
                                >
                                    {transformBusy ? <IconLoader size={16} className={styles.spinning} /> : <IconSparkles size={16} />}
                                    Apply (new version)
                                </button>
                            </div>
                        </div>

                        <div className={styles.transformGrid}>
                            <div className={styles.transformForm}>
                                <label className={styles.metaLabel} htmlFor="transformLabel">
                                    Version label (optional)
                                </label>
                                <input
                                    id="transformLabel"
                                    className={styles.input}
                                    value={transformLabel}
                                    onChange={(e) => setTransformLabel(e.target.value)}
                                    placeholder="e.g., dedup + fill_missing"
                                />

                                <label className={styles.checkboxRow}>
                                    <input
                                        type="checkbox"
                                        checked={transformSetCurrent}
                                        onChange={(e) => setTransformSetCurrent(e.target.checked)}
                                    />
                                    Set as current version when job completes
                                </label>

                                <div className={styles.modeHeader}>
                                    <div className={styles.metaLabel}>Steps</div>
                                    <div className={styles.modeTabs}>
                                        <button
                                            type="button"
                                            className={`${styles.modeTab} ${transformMode === 'builder' ? styles.modeTabActive : ''}`}
                                            onClick={() => (transformMode === 'json' ? switchToBuilder() : setTransformMode('builder'))}
                                            disabled={transformBusy || transformSuggestBusy || transformJobInFlight}
                                        >
                                            Builder
                                        </button>
                                        <button
                                            type="button"
                                            className={`${styles.modeTab} ${transformMode === 'json' ? styles.modeTabActive : ''}`}
                                            onClick={switchToJson}
                                            disabled={transformBusy || transformSuggestBusy || transformJobInFlight}
                                        >
                                            JSON
                                        </button>
                                    </div>
                                </div>

                                {transformMode === 'builder' ? (
                                    <TransformBuilder
                                        columns={dataset?.columns || []}
                                        steps={transformSteps}
                                        disabled={!ready || transformBusy || transformSuggestBusy || transformJobInFlight}
                                        onChange={(next, errs) => {
                                            setTransformSteps(Array.isArray(next) ? next : []);
                                            setTransformBuilderErrors(Array.isArray(errs) ? errs : []);
                                            try {
                                                setTransformStepsText(JSON.stringify(next || [], null, 2));
                                            } catch {
                                                // ignore
                                            }
                                        }}
                                    />
                                ) : (
                                    <>
                                        <textarea
                                            id="transformSteps"
                                            className={styles.textarea}
                                            value={transformStepsText}
                                            onChange={(e) => setTransformStepsText(e.target.value)}
                                            spellCheck={false}
                                            disabled={!ready || transformBusy || transformSuggestBusy || transformJobInFlight}
                                        />
                                        <div className={styles.muted}>
                                            Supported ops: <code>type_convert</code>, <code>fill_missing</code>, <code>drop_missing</code>,{' '}
                                            <code>deduplicate</code>, <code>string_normalize</code>, <code>drop_columns</code>,{' '}
                                            <code>rename_columns</code>, <code>filter_rows</code>, <code>sort_rows</code>, <code>limit_rows</code>,{' '}
                                            <code>time_features</code>, <code>bin_numeric</code>, <code>clip_outliers</code>,{' '}
                                            <code>encode_categorical</code>.
                                        </div>
                                    </>
                                )}
                            </div>

                            <div className={styles.transformOutput}>
                                {transformError ? <div className={styles.noticeError}>{transformError}</div> : null}
                                {transformJob ? (
                                    <div className={styles.noticeSuccess}>
                                        <div>{transformJob.message || 'Transformation job created'}</div>
                                        <div className={styles.mutedInline}>Job: {String(transformJob.job_id || '')}</div>
                                        {transformJobState ? (
                                            <div className={styles.muted} style={{ marginTop: 8 }}>
                                                <strong>Status:</strong> {String(transformJobState.status || '').toLowerCase() || 'unknown'}
                                                {Number.isFinite(Number(transformJobState.progress))
                                                    ? ` (${Math.round(Number(transformJobState.progress) * 100)}%)`
                                                    : ''}
                                                {transformJobState.status_message ? ` - ${String(transformJobState.status_message)}` : ''}
                                            </div>
                                        ) : null}
                                        <div style={{ marginTop: 10 }}>
                                            <Link href="/app/jobs" className={styles.secondaryBtn}>
                                                View Jobs
                                            </Link>
                                        </div>
                                    </div>
                                ) : null}

                                {Array.isArray(transformSuggestionPack?.suggestions) && transformSuggestionPack.suggestions.length ? (
                                    <div className={styles.pre} style={{ marginTop: 0 }}>
                                        <div style={{ fontWeight: 900, marginBottom: 8 }}>
                                            Suggested plan ({transformSuggestionPack.suggestions.length} step
                                            {transformSuggestionPack.suggestions.length === 1 ? '' : 's'})
                                        </div>
                                        <ol style={{ margin: 0, paddingLeft: 18 }}>
                                            {transformSuggestionPack.suggestions.map((item, idx) => (
                                                <li key={`sg-${idx}`} style={{ marginBottom: 6 }}>
                                                    <code>{String(item?.step?.op || 'unknown')}</code>
                                                    {item?.impact ? ` - ${String(item.impact)}` : ''}
                                                    <div className={styles.muted}>{String(item?.reason || '')}</div>
                                                </li>
                                            ))}
                                        </ol>
                                        {Array.isArray(transformSuggestionPack?.warnings) && transformSuggestionPack.warnings.length ? (
                                            <div className={styles.muted} style={{ marginTop: 8 }}>
                                                {transformSuggestionPack.warnings.join(' | ')}
                                            </div>
                                        ) : null}
                                    </div>
                                ) : null}

                                {transformPreview ? (
                                    <>
                                        <div className={styles.diffGrid}>
                                            <div className={styles.diffItem}>
                                                <div className={styles.metaLabel}>Rows</div>
                                                <div className={styles.metaValue}>
                                                    {Number(transformPreview.input_rows || 0).toLocaleString()} →{' '}
                                                    {Number(transformPreview.output_rows || 0).toLocaleString()}
                                                </div>
                                            </div>
                                            <div className={styles.diffItem}>
                                                <div className={styles.metaLabel}>Columns</div>
                                                <div className={styles.metaValue}>
                                                    {Number(transformPreview.input_columns || 0)} → {Number(transformPreview.output_columns || 0)}
                                                </div>
                                            </div>
                                        </div>

                                        {transformPreview?.added_columns?.length ? (
                                            <div className={styles.muted}>
                                                <strong>Added:</strong> {transformPreview.added_columns.slice(0, 12).join(', ')}
                                                {transformPreview.added_columns.length > 12 ? '…' : ''}
                                            </div>
                                        ) : null}
                                        {transformPreview?.removed_columns?.length ? (
                                            <div className={styles.muted}>
                                                <strong>Removed:</strong> {transformPreview.removed_columns.slice(0, 12).join(', ')}
                                                {transformPreview.removed_columns.length > 12 ? '…' : ''}
                                            </div>
                                        ) : null}
                                        {transformPreview?.warnings?.length ? (
                                            <pre className={styles.pre}>{safeJson(transformPreview.warnings)}</pre>
                                        ) : null}

                                        {transformPreview?.preview_rows?.length ? (
                                            <div className={styles.previewTableWrap} style={{ marginTop: 12 }}>
                                                <table className={styles.table}>
                                                    <thead>
                                                        <tr>
                                                            {Object.keys(transformPreview.preview_rows[0] || {}).map((k) => (
                                                                <th key={k}>{k}</th>
                                                            ))}
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {transformPreview.preview_rows.map((row, idx) => (
                                                            <tr key={idx}>
                                                                {Object.keys(transformPreview.preview_rows[0] || {}).map((k) => (
                                                                    <td key={k}>{String(row?.[k] ?? '')}</td>
                                                                ))}
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                            </div>
                                        ) : (
                                            <div className={styles.muted}>No preview rows returned.</div>
                                        )}

                                        {transformPreview?.metrics ? <pre className={styles.pre}>{safeJson(transformPreview.metrics)}</pre> : null}
                                    </>
                                ) : (
                                    <div className={styles.muted}>
                                        Preview runs on a bounded sample and shows a safe diff + output preview. Apply creates a new version via a
                                        background job.
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className={styles.section}>
                        <div className={styles.sectionHeader}>
                            <h2 className={styles.sectionTitle}>Schema</h2>
                            <input
                                className={styles.search}
                                value={colQuery}
                                onChange={(e) => setColQuery(e.target.value)}
                                placeholder="Search columns…"
                            />
                        </div>

                        <div className={styles.tableWrap}>
                            <table className={styles.table}>
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Column</th>
                                        <th>Type</th>
                                        <th>Null %</th>
                                        <th>Unique</th>
                                        <th>PII?</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredColumns.map((c) => (
                                        <tr key={c.name}>
                                            <td>{c.position}</td>
                                            <td title={c.original_name}>{c.name}</td>
                                            <td>{c.inferred_type}</td>
                                            <td>{(Number(c.null_percentage || 0) * 100).toFixed(2)}%</td>
                                            <td>{(c.unique_count || 0).toLocaleString()}</td>
                                            <td>{c.statistics?.is_potential_pii ? 'yes' : 'no'}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {(dataset?.quality_report?.warnings || []).length ? (
                        <div className={styles.section}>
                            <h2 className={styles.sectionTitle}>Warnings</h2>
                            <pre className={styles.pre}>{safeJson(dataset.quality_report.warnings)}</pre>
                        </div>
                    ) : null}
                </>
            ) : (
                <div className={styles.muted}>Dataset not found.</div>
            )}
        </div>
    );
}
