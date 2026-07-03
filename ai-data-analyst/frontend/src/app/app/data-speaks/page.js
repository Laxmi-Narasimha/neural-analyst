'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import ReactMarkdown from 'react-markdown';
import api from '@/lib/api';
import { IconDatabase, IconDownload, IconLoader, IconSparkles, IconTime, IconX } from '@/components/icons';
import styles from './page.module.css';

function safeJson(value) {
    try {
        return JSON.stringify(value, null, 2);
    } catch {
        return String(value);
    }
}

export default function DataSpeaksPage() {
    const searchParams = useSearchParams();
    const datasetParam = searchParams.get('dataset');

    const [datasets, setDatasets] = useState([]);
    const [selectedDatasetId, setSelectedDatasetId] = useState('');
    const [loadingDatasets, setLoadingDatasets] = useState(true);
    const [error, setError] = useState(null);

    const [analysisId, setAnalysisId] = useState(null);
    const [analysis, setAnalysis] = useState(null);
    const [creating, setCreating] = useState(false);
    const [downloading, setDownloading] = useState(null);
    const [streamFailed, setStreamFailed] = useState(false);
    const [exporting, setExporting] = useState(false);
    const [activePanel, setActivePanel] = useState('all');
    const [actionFeed, setActionFeed] = useState([]);
    const [actionBusy, setActionBusy] = useState(null);
    const pollersRef = useRef({});

    const selectedDataset = useMemo(
        () => datasets.find((d) => String(d.id) === String(selectedDatasetId)) || null,
        [datasets, selectedDatasetId]
    );

    const status = String(analysis?.status || '').toLowerCase();
    const running = status === 'queued' || status === 'running' || creating;
    const selectedStatus = String(selectedDataset?.status || '').toLowerCase();
    const canRun = Boolean(selectedDatasetId && selectedStatus === 'ready' && !running);

    const steps = useMemo(() => analysis?.results?.steps || [], [analysis]);
    const takeaways = useMemo(() => analysis?.results?.takeaways || [], [analysis]);
    const suggestedPrompts = useMemo(() => analysis?.results?.suggested_prompts || [], [analysis]);
    const suggestedActions = useMemo(() => analysis?.results?.suggested_actions || [], [analysis]);
    const runMeta = useMemo(() => analysis?.results?.run_meta || null, [analysis]);
    const narrativeMd = useMemo(() => String(analysis?.results?.narrative_md || ''), [analysis]);
    const insights = useMemo(() => analysis?.insights || [], [analysis]);
    const promptDatasetId = useMemo(
        () => String(selectedDatasetId || analysis?.dataset_id || ''),
        [analysis, selectedDatasetId]
    );
    const actionMap = useMemo(
        () => ({
            missingness_hotspots: 'missing',
            key_candidates: 'uniqueness',
            correlation_highlights: 'correlation',
            association_highlights: 'associations',
            outlier_columns: 'outliers',
            dominant_category: 'segments',
        }),
        []
    );

    const panelDefs = useMemo(
        () => [
            { id: 'all', label: 'All', ops: null },
            { id: 'overview', label: 'Overview', ops: new Set(['dataset_overview', 'preview_rows', 'numeric_summary', 'categorical_topk', 'text_summary']) },
            { id: 'schema', label: 'Schema', ops: new Set(['schema_snapshot']) },
            { id: 'risk', label: 'Privacy & Risk', ops: new Set(['privacy_risk_scan']) },
            { id: 'quality', label: 'Quality', ops: new Set(['missingness_scan', 'uniqueness_scan', 'outlier_scan']) },
            { id: 'outliers', label: 'Outliers', ops: new Set(['outlier_scan', 'outlier_explain']) },
            { id: 'relationships', label: 'Relationships', ops: new Set(['correlation_matrix', 'association_scan']) },
            { id: 'segments', label: 'Segments', ops: new Set(['segment_summary']) },
            { id: 'time', label: 'Time', ops: new Set(['resample_aggregate', 'time_anomaly_scan']) },
        ],
        []
    );

    const availablePanels = useMemo(() => {
        const opsPresent = new Set((steps || []).map((s) => String(s?.operator || '').trim()).filter(Boolean));
        return panelDefs.filter((p) => p.id === 'all' || (p.ops && Array.from(p.ops).some((op) => opsPresent.has(op))));
    }, [panelDefs, steps]);

    useEffect(() => {
        if (!availablePanels?.length) return;
        if (availablePanels.some((p) => p.id === activePanel)) return;
        setActivePanel(availablePanels[0].id);
    }, [activePanel, availablePanels]);

    const panelSteps = useMemo(() => {
        if (activePanel === 'all') return steps || [];
        const def = panelDefs.find((p) => p.id === activePanel);
        if (!def?.ops) return steps || [];
        return (steps || []).filter((s) => def.ops.has(String(s?.operator || '').trim()));
    }, [activePanel, panelDefs, steps]);

    const loadDatasets = useCallback(async () => {
        try {
            setLoadingDatasets(true);
            setError(null);
            const res = await api.listDatasets(1, 100);
            setDatasets(res.items || []);
        } catch (e) {
            setError(e?.message || 'Failed to load datasets');
        } finally {
            setLoadingDatasets(false);
        }
    }, []);

    useEffect(() => {
        loadDatasets();
    }, [loadDatasets]);

    useEffect(() => {
        if (datasetParam) {
            setSelectedDatasetId(datasetParam);
        }
    }, [datasetParam]);

    useEffect(() => {
        if (!selectedDatasetId && datasets?.[0]?.id) {
            setSelectedDatasetId(datasets[0].id);
        }
    }, [datasets, selectedDatasetId]);

    const loadAnalysis = useCallback(async () => {
        if (!analysisId) return;
        try {
            const a = await api.getAnalysis(analysisId);
            setAnalysis(a);
        } catch (e) {
            setError(e?.message || 'Failed to load analysis status');
        }
    }, [analysisId]);

    useEffect(() => {
        loadAnalysis();
    }, [loadAnalysis]);

    const loadActionFeed = useCallback(async () => {
        if (!analysisId) {
            setActionFeed([]);
            return;
        }
        try {
            const items = await api.listAnalysisActions(analysisId);
            setActionFeed(Array.isArray(items) ? items : []);
        } catch {
            // If the endpoint is unavailable, the parent analysis detail still includes action_feed.
        }
    }, [analysisId]);

    useEffect(() => {
        loadActionFeed();
    }, [loadActionFeed]);

    useEffect(() => {
        // New session: stop any in-flight action pollers.
        for (const key of Object.keys(pollersRef.current || {})) {
            stopPolling(key);
        }
        setActionFeed([]);
    }, [analysisId, stopPolling]);

    useEffect(() => {
        const local = analysis?.results?.action_feed;
        if (!analysisId) return;
        if (!Array.isArray(local) || local.length === 0) return;
        if (Array.isArray(actionFeed) && actionFeed.length > 0) return;
        setActionFeed(local);
    }, [analysisId, analysis, actionFeed]);

    useEffect(() => {
        if (!analysisId) return undefined;
        setStreamFailed(false);

        const stop = api.streamAnalysisEvents(analysisId, {
            onMeta: (evt) => {
                setAnalysis((prev) => {
                    const base = prev && String(prev.id) === String(analysisId) ? prev : prev || { id: analysisId };
                    const next = { ...base };

                    if (evt?.status != null) next.status = evt.status;
                    if (evt?.progress != null) next.progress = evt.progress;
                    if (evt?.status_message != null) next.status_message = evt.status_message;
                    if (evt?.started_at != null) next.started_at = evt.started_at;
                    if (evt?.completed_at != null) next.completed_at = evt.completed_at;
                    if (evt?.duration_seconds != null) next.duration_seconds = evt.duration_seconds;
                    if (evt?.error_message != null) next.error_message = evt.error_message;

                    const prevResults = (base?.results && typeof base.results === 'object') ? base.results : {};
                    const patch = evt?.results && typeof evt.results === 'object' ? evt.results : {};
                    const patchResults = patch?.run_meta != null || patch?.takeaways != null || patch?.suggested_prompts != null ? patch : {};

                    next.results = {
                        ...prevResults,
                        ...(patchResults.run_meta != null ? { run_meta: patchResults.run_meta } : {}),
                        ...(Array.isArray(patchResults.takeaways) ? { takeaways: patchResults.takeaways } : {}),
                        ...(Array.isArray(patchResults.suggested_prompts) ? { suggested_prompts: patchResults.suggested_prompts } : {}),
                        steps: Array.isArray(prevResults.steps) ? prevResults.steps : [],
                    };

                    return next;
                });
            },
            onStep: (evt) => {
                const step = evt?.step;
                if (!step || typeof step !== 'object') return;
                setAnalysis((prev) => {
                    const base = prev || { id: analysisId };
                    const prevResults = (base?.results && typeof base.results === 'object') ? base.results : {};
                    const prevSteps = Array.isArray(prevResults.steps) ? prevResults.steps : [];
                    const idx = Number(evt?.step_index) - 1;

                    const nextSteps = prevSteps.slice();
                    if (Number.isFinite(idx) && idx >= 0) nextSteps[idx] = step;
                    else nextSteps.push(step);

                    return {
                        ...base,
                        results: {
                            ...prevResults,
                            steps: nextSteps,
                        },
                    };
                });
            },
            onDone: () => {
                loadAnalysis();
            },
            onError: (e) => {
                setStreamFailed(true);
                setError(e?.message || 'Streaming failed; falling back to polling.');
            },
        });

        return () => stop && stop();
    }, [analysisId, loadAnalysis]);

    useEffect(() => {
        if (!analysisId) return undefined;
        if (!streamFailed) return undefined;
        if (status !== 'queued' && status !== 'running') return undefined;

        const id = setInterval(() => {
            loadAnalysis();
        }, 1500);
        return () => clearInterval(id);
    }, [analysisId, loadAnalysis, status, streamFailed]);

    const run = async () => {
        if (!selectedDatasetId) return;
        try {
            setCreating(true);
            setError(null);
            setAnalysisId(null);
            setAnalysis(null);

            const name = `Data Speaks: ${selectedDataset?.name || selectedDatasetId}`;
            const created = await api.createAnalysis(name, selectedDatasetId, 'eda', { sample_rows: 200_000 });
            setAnalysisId(created?.id || null);
            setAnalysis(created || null);
        } catch (e) {
            setError(e?.message || 'Failed to start Data Speaks');
        } finally {
            setCreating(false);
        }
    };

    const downloadArtifact = async (artifactId) => {
        try {
            setDownloading(String(artifactId));
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
        } finally {
            setDownloading(null);
        }
    };

    const exportReport = async () => {
        if (!analysisId) return;
        try {
            setExporting(true);
            setError(null);
            const artifact = await api.exportAnalysisReport(analysisId, 'markdown');
            if (artifact?.artifact_id) {
                await downloadArtifact(artifact.artifact_id);
            }
        } catch (e) {
            setError(e?.message || 'Export failed');
        } finally {
            setExporting(false);
        }
    };

    const collectArtifactsFromSteps = useCallback((stepsList) => {
        const out = [];
        const seen = new Set();
        if (!Array.isArray(stepsList)) return out;
        for (const s of stepsList) {
            if (!s || typeof s !== 'object') continue;
            const arts = Array.isArray(s.artifacts) ? s.artifacts : [];
            for (const a of arts) {
                if (!a || typeof a !== 'object') continue;
                const id = a.artifact_id != null ? String(a.artifact_id) : '';
                if (!id) continue;
                if (seen.has(id)) continue;
                seen.add(id);
                out.push(a);
                if (out.length >= 50) return out;
            }
        }
        return out;
    }, []);

    const upsertFeedItem = useCallback((item) => {
        const aid = item?.analysis_id != null ? String(item.analysis_id) : '';
        if (!aid) return;
        setActionFeed((prev) => {
            const arr = Array.isArray(prev) ? prev : [];
            const out = [];
            let inserted = false;
            for (const it of arr) {
                if (String(it?.analysis_id || '') === aid) {
                    out.push({ ...(it || {}), ...(item || {}) });
                    inserted = true;
                } else {
                    out.push(it);
                }
            }
            if (!inserted) out.unshift(item);
            return out;
        });
    }, []);

    const stopPolling = useCallback((childId) => {
        const key = String(childId || '');
        if (!key) return;
        const handle = pollersRef.current[key];
        if (handle) {
            try {
                clearInterval(handle);
            } catch {
                // ignore
            }
            delete pollersRef.current[key];
        }
    }, []);

    const startPolling = useCallback(
        (childId) => {
            const key = String(childId || '');
            if (!key) return;
            if (pollersRef.current[key]) return;

            const tick = async () => {
                try {
                    const a = await api.getAnalysis(key);
                    const st = String(a?.status || '').toLowerCase();
                    const stepsList = a?.results?.steps;
                    const artifacts = collectArtifactsFromSteps(stepsList);
                    const take = Array.isArray(a?.results?.takeaways) ? a.results.takeaways : [];
                    upsertFeedItem({
                        analysis_id: key,
                        status: a?.status,
                        status_message: a?.status_message,
                        error_message: a?.error_message || null,
                        takeaways: take,
                        artifacts,
                    });

                    if (st && st !== 'queued' && st !== 'running') {
                        stopPolling(key);
                        loadActionFeed();
                    }
                } catch {
                    // Ignore transient failures (offline / server restart).
                }
            };

            tick();
            const id = setInterval(tick, 1500);
            pollersRef.current[key] = id;
        },
        [collectArtifactsFromSteps, loadActionFeed, stopPolling, upsertFeedItem]
    );

    useEffect(() => {
        const arr = Array.isArray(actionFeed) ? actionFeed : [];
        for (const it of arr) {
            const st = String(it?.status || '').toLowerCase();
            const cid = it?.analysis_id != null ? String(it.analysis_id) : '';
            if (!cid) continue;
            if (st === 'queued' || st === 'running') startPolling(cid);
        }
    }, [actionFeed, startPolling]);

    useEffect(() => {
        return () => {
            for (const key of Object.keys(pollersRef.current || {})) {
                stopPolling(key);
            }
        };
    }, [stopPolling]);

    const runSuggestedAction = async (action) => {
        if (!analysisId) return;
        const actionId = String(action?.action_id || action?.actionId || '').trim();
        if (!actionId) return;

        const kind = String(action?.kind || 'analysis').trim().toLowerCase();
        if (kind === 'report' && actionId === 'export_report') {
            exportReport();
            return;
        }

        if (kind !== 'analysis') return;

        try {
            setActionBusy(actionId);
            setError(null);
            const params = action?.params && typeof action.params === 'object' ? action.params : {};
            const item = await api.runAnalysisAction(analysisId, actionId, params);
            upsertFeedItem(item);
            startPolling(item?.analysis_id);
        } catch (e) {
            setError(e?.message || 'Failed to run action');
        } finally {
            setActionBusy(null);
        }
    };

    return (
        <div className={styles.page}>
            <div className={styles.header}>
                <div className={styles.titleRow}>
                    <div className={styles.titleIcon}>
                        <IconSparkles size={18} />
                    </div>
                    <div>
                        <h1 className={styles.title}>Data Speaks</h1>
                        <p className={styles.subtitle}>
                            One-click EDA run (persisted) with safe operators, evidence artifacts, and progress.
                        </p>
                    </div>
                </div>

                <div className={styles.controls}>
                    <div className={styles.selector}>
                        <IconDatabase size={16} />
                        <select
                            className={styles.select}
                            value={selectedDatasetId}
                            onChange={(e) => setSelectedDatasetId(e.target.value)}
                            disabled={loadingDatasets || running}
                        >
                            <option value="" disabled>Select a dataset</option>
                            {datasets.map((d) => (
                                <option key={d.id} value={d.id}>
                                    {d.name} ({d.status})
                                </option>
                            ))}
                        </select>
                    </div>

                    <button className={styles.runBtn} onClick={run} disabled={!canRun}>
                        {running ? <IconLoader size={16} className={styles.spinning} /> : <IconSparkles size={16} />}
                        {running ? 'Running...' : 'Make the data speak'}
                    </button>

                    {analysisId && steps?.length ? (
                        <button className={styles.exportBtn} onClick={exportReport} disabled={exporting} type="button">
                            {exporting ? <IconLoader size={16} className={styles.spinning} /> : <IconDownload size={16} />}
                            Export
                        </button>
                    ) : null}
                </div>
            </div>

            {error && (
                <motion.div className={styles.error} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
                    <IconX size={18} />
                    {error}
                    <button className={styles.dismiss} onClick={() => setError(null)} type="button">
                        Dismiss
                    </button>
                </motion.div>
            )}

            {selectedDataset && (
                <div className={styles.datasetMeta}>
                    <div><strong>Dataset:</strong> {selectedDataset.name}</div>
                    <div><strong>Status:</strong> {selectedDataset.status}</div>
                    <div><strong>Rows:</strong> {(selectedDataset.row_count || 0).toLocaleString()}</div>
                    <div><strong>Columns:</strong> {selectedDataset.column_count || 0}</div>
                    {selectedStatus !== 'ready' && (
                        <div className={styles.muted}>Dataset is not ready yet. Wait for processing to finish.</div>
                    )}
                </div>
            )}

            {analysis && (
                <div className={styles.datasetMeta}>
                    <div><strong>Analysis:</strong> {analysis.name}</div>
                    <div><strong>Status:</strong> {analysis.status} {analysis.progress != null ? `(${Math.round(analysis.progress * 100)}%)` : ''}</div>
                    {analysis.status_message ? <div><strong>Message:</strong> {analysis.status_message}</div> : null}
                    <div className={styles.muted}>
                        <Link href={`/app/analysis/${analysis.id}`}>Open analysis detail</Link>
                    </div>
                </div>
            )}

            {(analysis && (takeaways?.length || suggestedPrompts?.length || runMeta)) ? (
                <div className={styles.summaryCard}>
                    <div className={styles.summaryHeader}>
                        <div className={styles.summaryTitle}>Top takeaways</div>
                        {runMeta?.confidence ? (
                            <div className={styles.muted}>
                                <strong>Confidence:</strong> {String(runMeta.confidence)}
                            </div>
                        ) : null}
                    </div>

                    {takeaways?.length ? (
                        <div className={styles.takeaways}>
                            {takeaways.map((t, idx) => (
                                <div key={`${idx}-${String(t).slice(0, 24)}`} className={styles.takeawayItem}>
                                    <span className={styles.takeawayIndex}>{idx + 1}</span>
                                    <span>{String(t)}</span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className={styles.muted}>No takeaways generated yet.</div>
                    )}

                    {narrativeMd ? (
                        <div className={styles.narrative}>
                            <div className={styles.promptsTitle}>Narrative</div>
                            <div className={styles.markdown}>
                                <ReactMarkdown>{narrativeMd}</ReactMarkdown>
                            </div>
                        </div>
                    ) : null}

                    {suggestedPrompts?.length ? (
                        <div className={styles.prompts}>
                            <div className={styles.promptsTitle}>Suggested prompts</div>
                            <div className={styles.promptList}>
                                {suggestedPrompts.map((p) => {
                                    const text = String(p || '').trim();
                                    if (!text) return null;
                                    const href = `/app/analysis/new?dataset=${encodeURIComponent(promptDatasetId)}&prompt=${encodeURIComponent(text)}&autoSend=1`;
                                    return (
                                        <Link key={text} href={href} className={styles.promptItem}>
                                            {text}
                                        </Link>
                                    );
                                })}
                            </div>
                        </div>
                    ) : null}

                    {insights?.length ? (
                        <div className={styles.insightLibrary}>
                            <div className={styles.promptsTitle}>Insight library</div>
                            <div className={styles.insightGrid}>
                                {insights.slice(0, 10).map((ins, idx) => {
                                    const score = Number(ins?.score || 0);
                                    const scorePct = Number.isFinite(score) ? Math.round(Math.max(0, Math.min(1, score)) * 100) : 0;
                                    const aids = Array.isArray(ins?.artifact_ids) ? ins.artifact_ids : [];
                                    const kind = String(ins?.kind || '').trim();
                                    const actionId = actionMap[kind] || null;
                                    const runHref =
                                        actionId && promptDatasetId
                                            ? `/app/analysis/new?dataset=${encodeURIComponent(promptDatasetId)}&action=${encodeURIComponent(actionId)}&autoRun=1`
                                            : null;
                                    return (
                                        <div key={`${ins?.kind || 'ins'}-${idx}`} className={styles.insightCard}>
                                            <div className={styles.insightTop}>
                                                <div className={styles.insightTopLeft}>
                                                    <span className={styles.badge}>{String(ins?.kind || 'insight')}</span>
                                                    <span className={styles.scorePill}>{scorePct}%</span>
                                                </div>
                                                {(runHref || aids.length) ? (
                                                    <div className={styles.insightArtifacts}>
                                                        {runHref ? (
                                                            <Link href={runHref} className={styles.insightRun}>
                                                                Run
                                                            </Link>
                                                        ) : null}
                                                        {aids.slice(0, 3).map((aid) => (
                                                            <button
                                                                key={aid}
                                                                className={styles.evidenceBtn}
                                                                onClick={() => downloadArtifact(aid)}
                                                                type="button"
                                                                title="Download evidence artifact"
                                                                disabled={downloading === String(aid)}
                                                            >
                                                                {downloading === String(aid) ? (
                                                                    <IconLoader size={14} className={styles.spinning} />
                                                                ) : (
                                                                    <IconDownload size={14} />
                                                                )}
                                                            </button>
                                                        ))}
                                                    </div>
                                                ) : null}
                                            </div>
                                            <div className={styles.insightTitle}>{String(ins?.title || 'Insight')}</div>
                                            <div className={styles.insightDetail}>{String(ins?.detail || '')}</div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    ) : null}
                </div>
            ) : null}

            {(analysis && Array.isArray(suggestedActions) && suggestedActions.length) ? (
                <div className={styles.actionsCard}>
                    <div className={styles.actionsHeader}>
                        <div className={styles.actionsTitle}>Suggested next actions</div>
                        <div className={styles.muted}>One click. Evidence-backed. Logged in the action feed.</div>
                    </div>

                    <div className={styles.actionsGrid}>
                        {suggestedActions.slice(0, 10).map((a) => {
                            const actionId = String(a?.action_id || '').trim();
                            if (!actionId) return null;
                            const kind = String(a?.kind || 'analysis').trim().toLowerCase();
                            const title = String(a?.title || actionId);
                            const detail = a?.detail != null ? String(a.detail) : '';

                            if (kind === 'navigate' && actionId === 'open_sql') {
                                const href = promptDatasetId ? `/app/datasets/${encodeURIComponent(promptDatasetId)}#sql` : '/app/datasets';
                                return (
                                    <Link key={`${kind}-${actionId}`} href={href} className={styles.actionBtn}>
                                        <div className={styles.actionBtnTitle}>{title}</div>
                                        {detail ? <div className={styles.actionBtnDetail}>{detail}</div> : null}
                                    </Link>
                                );
                            }

                            if (kind === 'navigate' && (actionId === 'transform_fill_missing' || actionId === 'transform_deduplicate')) {
                                const params = a?.params && typeof a.params === 'object' ? a.params : {};
                                const qp = new URLSearchParams();
                                if (actionId === 'transform_fill_missing') {
                                    qp.set('transformPreset', 'fill_missing');
                                    const cols = Array.isArray(params.columns) ? params.columns : [];
                                    if (cols.length) qp.set('columns', cols.map((c) => String(c)).join(','));
                                } else {
                                    qp.set('transformPreset', 'deduplicate');
                                    const subset = Array.isArray(params.subset) ? params.subset : [];
                                    if (subset.length) qp.set('subset', subset.map((c) => String(c)).join(','));
                                }
                                const base = promptDatasetId ? `/app/datasets/${encodeURIComponent(promptDatasetId)}` : '/app/datasets';
                                const href = `${base}?${qp.toString()}#transform`;
                                return (
                                    <Link key={`${kind}-${actionId}`} href={href} className={styles.actionBtn}>
                                        <div className={styles.actionBtnTitle}>{title}</div>
                                        {detail ? <div className={styles.actionBtnDetail}>{detail}</div> : null}
                                    </Link>
                                );
                            }

                            const disabled =
                                running ||
                                (kind === 'report' ? exporting : false) ||
                                (kind === 'analysis' ? Boolean(actionBusy) : false);

                            return (
                                <button
                                    key={`${kind}-${actionId}`}
                                    type="button"
                                    className={styles.actionBtn}
                                    onClick={() => runSuggestedAction(a)}
                                    disabled={disabled}
                                    title={detail || title}
                                >
                                    <div className={styles.actionBtnTitle}>
                                        {kind === 'analysis' && actionBusy === actionId ? (
                                            <IconLoader size={14} className={styles.spinning} />
                                        ) : null}
                                        {title}
                                    </div>
                                    {detail ? <div className={styles.actionBtnDetail}>{detail}</div> : null}
                                </button>
                            );
                        })}
                    </div>
                </div>
            ) : null}

            {analysis ? (
                <div className={styles.feedCard}>
                    <div className={styles.actionsHeader}>
                        <div className={styles.actionsTitle}>Action feed</div>
                        <div className={styles.muted}>Every action run is tracked with outputs and a re-run button.</div>
                    </div>

                    {Array.isArray(actionFeed) && actionFeed.length ? (
                        <div className={styles.feedList}>
                            {actionFeed.map((it, idx) => {
                                const childId = it?.analysis_id != null ? String(it.analysis_id) : '';
                                const title = String(it?.title || it?.action_id || `Action ${idx + 1}`);
                                const detail = it?.detail != null ? String(it.detail) : '';
                                const st = String(it?.status || '').toLowerCase();
                                const statusLabel = st || 'queued';
                                const createdAt = it?.created_at ? new Date(it.created_at).toLocaleString() : '';
                                const take = Array.isArray(it?.takeaways) ? it.takeaways : [];
                                const artifacts = Array.isArray(it?.artifacts) ? it.artifacts : [];
                                const params = it?.params && typeof it.params === 'object' ? it.params : {};

                                return (
                                    <div key={`${childId || idx}`} className={styles.feedItem}>
                                        <div className={styles.feedTop}>
                                            <div className={styles.feedTitleRow}>
                                                <div className={styles.feedTitle}>{title}</div>
                                                <span className={`${styles.statusPill} ${st === 'completed' ? styles.statusOk : ''} ${st === 'failed' ? styles.statusBad : ''}`}>
                                                    {statusLabel}
                                                </span>
                                            </div>
                                            <div className={styles.feedMeta}>
                                                {createdAt ? <span>{createdAt}</span> : null}
                                                {childId ? (
                                                    <Link href={`/app/analysis/${encodeURIComponent(childId)}`} className={styles.feedLink}>
                                                        Open
                                                    </Link>
                                                ) : null}
                                                <button
                                                    type="button"
                                                    className={styles.feedRerun}
                                                    onClick={() => runSuggestedAction({ action_id: it?.action_id, kind: 'analysis', params })}
                                                    disabled={running || Boolean(actionBusy)}
                                                >
                                                    Re-run
                                                </button>
                                            </div>
                                            {detail ? <div className={styles.feedDetail}>{detail}</div> : null}
                                            {it?.error_message ? <div className={styles.feedError}>{String(it.error_message)}</div> : null}
                                        </div>

                                        {take.length ? (
                                            <div className={styles.feedTakeaways}>
                                                {take.slice(0, 5).map((t, i) => (
                                                    <div key={`${childId}-t-${i}`} className={styles.takeawayItem}>
                                                        <span className={styles.takeawayIndex}>{i + 1}</span>
                                                        <span>{String(t)}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : null}

                                        {artifacts.length ? (
                                            <div className={styles.feedArtifacts}>
                                                {artifacts.slice(0, 8).map((a, aidx) => (
                                                    <div key={String(a?.artifact_id || `${childId}-a-${aidx}`)} className={styles.artifact}>
                                                        <div className={styles.artifactHeader}>
                                                            <div className={styles.artifactTitle}>
                                                                <span className={styles.badge}>{a?.artifact_type}</span>
                                                                <Link
                                                                    href={`/app/artifacts/${encodeURIComponent(String(a?.artifact_id || ''))}`}
                                                                    className={styles.artifactName}
                                                                    title="Open artifact viewer"
                                                                >
                                                                    {a?.name || String(a?.artifact_id || '')}
                                                                </Link>
                                                            </div>
                                                            {a?.artifact_id ? (
                                                                <button
                                                                    className={styles.downloadBtn}
                                                                    onClick={() => downloadArtifact(a.artifact_id)}
                                                                    title="Download artifact data"
                                                                    type="button"
                                                                    disabled={downloading === String(a.artifact_id)}
                                                                >
                                                                    {downloading === String(a.artifact_id) ? (
                                                                        <IconLoader size={16} className={styles.spinning} />
                                                                    ) : (
                                                                        <IconDownload size={16} />
                                                                    )}
                                                                </button>
                                                            ) : null}
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : null}
                                    </div>
                                );
                            })}
                        </div>
                    ) : (
                        <div className={styles.muted}>No actions run yet.</div>
                    )}
                </div>
            ) : null}

            {!analysis && !running && (
                <div className={styles.empty}>
                    <p>Select a dataset and run Data Speaks to generate evidence artifacts.</p>
                </div>
            )}

            {steps?.length ? (
                <div className={styles.steps}>
                    <div className={styles.panelTabs}>
                        {availablePanels.map((p) => (
                            <button
                                key={p.id}
                                type="button"
                                className={`${styles.panelTab} ${activePanel === p.id ? styles.panelTabActive : ''}`}
                                onClick={() => setActivePanel(p.id)}
                            >
                                {p.label}
                            </button>
                        ))}
                    </div>

                    {panelSteps.map((step, idx) => (
                        <div key={`${step.operator}-${idx}`} className={styles.stepCard}>
                            <div className={styles.stepHeader}>
                                <div className={styles.stepTitle}>
                                    <span className={styles.stepIndex}>{idx + 1}</span>
                                    <span>{step.operator}</span>
                                </div>
                                <div className={styles.muted}>
                                    <IconTime size={14} /> step
                                </div>
                            </div>

                            {step.summary && <pre className={styles.pre}>{safeJson(step.summary)}</pre>}

                            {step.artifacts?.length ? (
                                <div className={styles.artifacts}>
                                    {step.artifacts.map((a) => (
                                        <div key={a.artifact_id} className={styles.artifact}>
                                            <div className={styles.artifactHeader}>
                                                    <div className={styles.artifactTitle}>
                                                        <span className={styles.badge}>{a.artifact_type}</span>
                                                        <Link
                                                            href={`/app/artifacts/${encodeURIComponent(String(a.artifact_id))}`}
                                                            className={styles.artifactName}
                                                            title="Open artifact viewer"
                                                        >
                                                            {a.name}
                                                        </Link>
                                                    </div>
                                                    <button
                                                        className={styles.downloadBtn}
                                                        onClick={() => downloadArtifact(a.artifact_id)}
                                                        title="Download artifact data"
                                                        type="button"
                                                        disabled={downloading === String(a.artifact_id)}
                                                    >
                                                        {downloading === String(a.artifact_id) ? (
                                                            <IconLoader size={16} className={styles.spinning} />
                                                        ) : (
                                                            <IconDownload size={16} />
                                                        )}
                                                    </button>
                                            </div>

                                            {a.preview && <pre className={styles.preSmall}>{safeJson(a.preview)}</pre>}
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className={styles.muted}>No artifacts produced.</div>
                            )}
                        </div>
                    ))}
                </div>
            ) : null}
        </div>
    );
}
