'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { motion } from 'framer-motion';
import api from '@/lib/api';
import {
    IconArrowRight,
    IconChart,
    IconDownload,
    IconLoader,
    IconTime,
    IconX,
} from '@/components/icons';
import styles from './page.module.css';

function safeJson(value) {
    try {
        return JSON.stringify(value, null, 2);
    } catch {
        return String(value);
    }
}

function formatDateTime(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    return d.toLocaleString();
}

export default function AnalysisDetailPage() {
    const params = useParams();
    const analysisId = params?.analysisId;

    const [analysis, setAnalysis] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [downloading, setDownloading] = useState(null);
    const [cancelling, setCancelling] = useState(false);
    const [streamFailed, setStreamFailed] = useState(false);
    const [exporting, setExporting] = useState(false);
    const [reportArtifact, setReportArtifact] = useState(null);

    const status = String(analysis?.status || '').toLowerCase();
    const steps = useMemo(() => analysis?.results?.steps || [], [analysis]);
    const takeaways = useMemo(() => analysis?.results?.takeaways || [], [analysis]);
    const suggestedPrompts = useMemo(() => analysis?.results?.suggested_prompts || [], [analysis]);
    const runMeta = useMemo(() => analysis?.results?.run_meta || null, [analysis]);
    const insights = useMemo(() => analysis?.insights || [], [analysis]);

    const load = useCallback(async () => {
        if (!analysisId) return;
        try {
            setError(null);
            const res = await api.getAnalysis(analysisId);
            setAnalysis(res);
        } catch (e) {
            setError(e?.message || 'Failed to load analysis');
        } finally {
            setLoading(false);
        }
    }, [analysisId]);

    useEffect(() => {
        load();
    }, [load]);

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
                setLoading(false);
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
                load();
            },
            onError: (e) => {
                setStreamFailed(true);
                setError(e?.message || 'Streaming failed; falling back to polling.');
            },
        });

        return () => stop && stop();
    }, [analysisId, load]);

    useEffect(() => {
        if (!analysisId) return undefined;
        if (!streamFailed) return undefined;
        if (status !== 'queued' && status !== 'running') return undefined;

        const id = setInterval(() => {
            load();
        }, 1500);

        return () => clearInterval(id);
    }, [analysisId, load, status, streamFailed]);

    const downloadArtifact = async (artifactId) => {
        if (!artifactId) return;
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

    const cancel = async () => {
        if (!analysisId) return;
        try {
            setCancelling(true);
            setError(null);
            await api.cancelAnalysis(analysisId);
            await load();
        } catch (e) {
            setError(e?.message || 'Cancel failed');
        } finally {
            setCancelling(false);
        }
    };

    const exportReport = async (format) => {
        if (!analysisId) return;
        try {
            setExporting(true);
            setError(null);
            const artifact = await api.exportAnalysisReport(analysisId, format);
            setReportArtifact(artifact || null);
            if (artifact?.artifact_id) {
                await downloadArtifact(artifact.artifact_id);
            }
        } catch (e) {
            setError(e?.message || 'Export failed');
        } finally {
            setExporting(false);
        }
    };

    return (
        <div className={styles.page}>
            <div className={styles.header}>
                <div className={styles.headerLeft}>
                    <Link href="/app/analysis" className={styles.back}>
                        <IconArrowRight size={18} className={styles.backIcon} />
                        Back
                    </Link>
                    <div className={styles.titleRow}>
                        <div className={styles.titleIcon}>
                            <IconChart size={18} />
                        </div>
                        <div>
                            <h1 className={styles.title}>{analysis?.name || 'Analysis'}</h1>
                            <div className={styles.subtitle}>
                                <span className={`${styles.badge} ${styles[status]}`}>{status || 'unknown'}</span>
                                {analysis?.status_message ? <span>{analysis.status_message}</span> : null}
                            </div>
                        </div>
                    </div>
                </div>

                <div className={styles.headerRight}>
                    {(status === 'queued' || status === 'running') && (
                        <button className={styles.cancelBtn} onClick={cancel} type="button" disabled={cancelling}>
                            {cancelling ? <IconLoader size={16} className={styles.spinning} /> : <IconX size={16} />}
                            Cancel
                        </button>
                    )}
                    <button
                        className={styles.exportBtn}
                        onClick={() => exportReport('markdown')}
                        type="button"
                        disabled={exporting || !steps?.length}
                        title={steps?.length ? 'Export grounded markdown report' : 'No steps yet to export'}
                    >
                        {exporting ? <IconLoader size={16} className={styles.spinning} /> : <IconDownload size={16} />}
                        Export
                    </button>
                    <button className={styles.refreshBtn} onClick={load} type="button" disabled={loading}>
                        {loading ? <IconLoader size={16} className={styles.spinning} /> : <IconTime size={16} />}
                        Refresh
                    </button>
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

            {reportArtifact?.artifact_id ? (
                <div className={styles.muted}>
                    Latest report artifact: <strong>{String(reportArtifact.artifact_id)}</strong>
                </div>
            ) : null}

            <div className={styles.metaGrid}>
                <div className={styles.metaCard}>
                    <div className={styles.metaLabel}>Analysis ID</div>
                    <div className={styles.metaValue}>{String(analysis?.id || analysisId || '')}</div>
                </div>
                <div className={styles.metaCard}>
                    <div className={styles.metaLabel}>Dataset</div>
                    <div className={styles.metaValue}>{String(analysis?.dataset_id || '')}</div>
                </div>
                <div className={styles.metaCard}>
                    <div className={styles.metaLabel}>Created</div>
                    <div className={styles.metaValue}>{formatDateTime(analysis?.created_at)}</div>
                </div>
                <div className={styles.metaCard}>
                    <div className={styles.metaLabel}>Runtime</div>
                    <div className={styles.metaValue}>
                        {analysis?.duration_seconds != null ? `${Math.round(analysis.duration_seconds)}s` : '-'}
                    </div>
                </div>
            </div>

            {analysis?.error_message ? (
                <div className={styles.section}>
                    <h2 className={styles.sectionTitle}>Error</h2>
                    <pre className={styles.pre}>{safeJson({ error: analysis.error_message })}</pre>
                </div>
            ) : null}

            {(takeaways?.length || suggestedPrompts?.length || runMeta) ? (
                <div className={styles.section}>
                    <h2 className={styles.sectionTitle}>Summary</h2>

                    {runMeta ? (
                        <div className={styles.muted}>
                            {runMeta.confidence ? <span><strong>Confidence:</strong> {String(runMeta.confidence)}</span> : null}
                            {runMeta.scanned_rows != null ? (
                                <span>{' '}<strong>Scanned:</strong> {Number(runMeta.scanned_rows).toLocaleString()} rows</span>
                            ) : null}
                            {runMeta.dataset_rows != null ? (
                                <span>{' '}<strong>Total:</strong> {Number(runMeta.dataset_rows).toLocaleString()} rows</span>
                            ) : null}
                        </div>
                    ) : null}

                    {takeaways?.length ? (
                        <div className={styles.takeaways}>
                            {takeaways.map((t, idx) => (
                                <div key={`${idx}-${String(t).slice(0, 20)}`} className={styles.takeawayItem}>
                                    <span className={styles.takeawayIndex}>{idx + 1}</span>
                                    <span>{String(t)}</span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className={styles.muted}>No takeaways generated.</div>
                    )}

                    {suggestedPrompts?.length ? (
                        <div className={styles.prompts}>
                            <div className={styles.promptsTitle}>Suggested prompts</div>
                            <div className={styles.promptList}>
                                {suggestedPrompts.map((p) => (
                                    <div key={p} className={styles.promptItem}>{p}</div>
                                ))}
                            </div>
                        </div>
                    ) : null}

                    {insights?.length ? (
                        <div className={styles.insightLibrary}>
                            <div className={styles.promptsTitle}>Insight library</div>
                            <div className={styles.insightGrid}>
                                {insights.slice(0, 12).map((ins, idx) => {
                                    const score = Number(ins?.score || 0);
                                    const scorePct = Number.isFinite(score) ? Math.round(Math.max(0, Math.min(1, score)) * 100) : 0;
                                    const aids = Array.isArray(ins?.artifact_ids) ? ins.artifact_ids : [];
                                    return (
                                        <div key={`${ins?.kind || 'ins'}-${idx}`} className={styles.insightCard}>
                                            <div className={styles.insightTop}>
                                                <div className={styles.insightTopLeft}>
                                                    <span className={styles.insightKind}>{String(ins?.kind || 'insight')}</span>
                                                    <span className={styles.scorePill}>{scorePct}%</span>
                                                </div>
                                                {aids.length ? (
                                                    <div className={styles.insightArtifacts}>
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

            <div className={styles.section}>
                <h2 className={styles.sectionTitle}>Steps</h2>

                {!steps?.length ? (
                    <div className={styles.muted}>
                        {status === 'queued' || status === 'running'
                            ? 'No steps completed yet. This page will update as the run progresses.'
                            : 'No steps recorded for this analysis.'}
                    </div>
                ) : (
                    <div className={styles.steps}>
                        {steps.map((step, idx) => (
                            <div key={`${step.operator}-${idx}`} className={styles.stepCard}>
                                <div className={styles.stepHeader}>
                                    <div className={styles.stepTitle}>
                                        <span className={styles.stepIndex}>{idx + 1}</span>
                                        <span>{step.operator}</span>
                                    </div>
                                </div>

                                {step.summary ? <pre className={styles.pre}>{safeJson(step.summary)}</pre> : null}

                                {step.artifacts?.length ? (
                                    <div className={styles.artifacts}>
                                        {step.artifacts.map((a) => (
                                            <div key={a.artifact_id} className={styles.artifact}>
                                                <div className={styles.artifactHeader}>
                                                    <div className={styles.artifactTitle}>
                                                        <span className={styles.artifactBadge}>{a.artifact_type}</span>
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
                                                        type="button"
                                                        disabled={downloading === String(a.artifact_id)}
                                                        title="Download artifact data"
                                                    >
                                                        {downloading === String(a.artifact_id) ? (
                                                            <IconLoader size={16} className={styles.spinning} />
                                                        ) : (
                                                            <IconDownload size={16} />
                                                        )}
                                                    </button>
                                                </div>

                                                {a.preview ? (
                                                    <pre className={styles.preSmall}>{safeJson(a.preview)}</pre>
                                                ) : (
                                                    <div className={styles.muted}>No preview available.</div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div className={styles.muted}>No artifacts produced.</div>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
