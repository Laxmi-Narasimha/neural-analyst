'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import api from '@/lib/api';
import { IconLoader, IconTime, IconX } from '@/components/icons';
import styles from './page.module.css';

function formatDateTime(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString();
}

function clamp01(x) {
    const n = Number(x);
    if (!Number.isFinite(n)) return 0;
    return Math.max(0, Math.min(1, n));
}

export default function JobsPage() {
    const [jobs, setJobs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [filter, setFilter] = useState(''); // status filter
    const [cancellingId, setCancellingId] = useState(null);

    const load = useCallback(async () => {
        try {
            setLoading(true);
            setError(null);
            const res = await api.listJobs(1, 100, filter ? { status: filter } : {});
            setJobs(res.items || []);
        } catch (e) {
            setError(e?.message || 'Failed to load jobs');
            setJobs([]);
        } finally {
            setLoading(false);
        }
    }, [filter]);

    useEffect(() => {
        load();
    }, [load]);

    const cancelJob = useCallback(
        async (jobId) => {
            if (!jobId) return;
            try {
                setCancellingId(String(jobId));
                setError(null);
                await api.cancelJob(jobId);
                await load();
            } catch (e) {
                setError(e?.message || 'Failed to cancel job');
            } finally {
                setCancellingId(null);
            }
        },
        [load]
    );

    const sorted = useMemo(() => {
        const arr = Array.isArray(jobs) ? jobs.slice() : [];
        arr.sort((a, b) => String(b?.created_at || '').localeCompare(String(a?.created_at || '')));
        return arr;
    }, [jobs]);

    return (
        <div className={styles.page}>
            <div className={styles.header}>
                <div className={styles.titleRow}>
                    <div className={styles.titleIcon}><IconTime size={18} /></div>
                    <div>
                        <h1 className={styles.title}>Jobs</h1>
                        <p className={styles.subtitle}>Background tasks (dataset processing, async compute).</p>
                    </div>
                </div>

                <div className={styles.controls}>
                    <select
                        className={styles.select}
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                        disabled={loading}
                    >
                        <option value="">All statuses</option>
                        <option value="queued">Queued</option>
                        <option value="running">Running</option>
                        <option value="completed">Completed</option>
                        <option value="failed">Failed</option>
                        <option value="cancelled">Cancelled</option>
                    </select>

                    <button className={styles.refreshBtn} onClick={load} disabled={loading}>
                        {loading ? <IconLoader size={16} className={styles.spinning} /> : 'Refresh'}
                    </button>
                </div>
            </div>

            {error && (
                <motion.div className={styles.error} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
                    <IconX size={18} />
                    {error}
                    <button className={styles.dismiss} onClick={() => setError(null)}>Dismiss</button>
                </motion.div>
            )}

            {loading ? (
                <div className={styles.loading}>
                    <IconLoader size={32} className={styles.spinning} />
                    <p>Loading jobs...</p>
                </div>
            ) : (
                <div className={styles.list}>
                    {sorted.length === 0 ? (
                        <div className={styles.empty}>No jobs found.</div>
                    ) : (
                        sorted.map((j) => {
                            const status = String(j?.status || '').toLowerCase();
                            const progress = clamp01(j?.progress);
                            const pct = Math.round(progress * 100);
                            const canCancel = status === 'queued' || status === 'running';
                            return (
                                <div key={j.id} className={styles.card}>
                                    <div className={styles.cardTop}>
                                        <div className={styles.mainInfo}>
                                            <div className={styles.badges}>
                                                <span className={styles.badge}>{j.job_type}</span>
                                                <span className={`${styles.badge} ${styles['status_' + status] || ''}`}>{status || 'unknown'}</span>
                                                {canCancel && (
                                                    <button
                                                        className={styles.cancelBtn}
                                                        onClick={() => cancelJob(j.id)}
                                                        type="button"
                                                        disabled={cancellingId === String(j.id)}
                                                        title="Cancel job"
                                                    >
                                                        {cancellingId === String(j.id) ? (
                                                            <IconLoader size={14} className={styles.spinning} />
                                                        ) : (
                                                            <IconX size={14} />
                                                        )}
                                                    </button>
                                                )}
                                            </div>
                                            <div className={styles.meta}>
                                                <div><strong>Job:</strong> {j.id}</div>
                                                {j.dataset_id && <div><strong>Dataset:</strong> {j.dataset_id}</div>}
                                            </div>
                                        </div>
                                        <div className={styles.timeCol}>
                                            <div className={styles.time}><strong>Created:</strong> {formatDateTime(j.created_at)}</div>
                                            {j.completed_at && <div className={styles.time}><strong>Done:</strong> {formatDateTime(j.completed_at)}</div>}
                                        </div>
                                    </div>

                                    <div className={styles.progressRow}>
                                        <div className={styles.progressBar} aria-label={`Progress ${pct}%`}>
                                            <div className={styles.progressFill} style={{ width: `${pct}%` }} />
                                        </div>
                                        <div className={styles.progressText}>{pct}%</div>
                                    </div>

                                    {j.status_message && <div className={styles.message}>{j.status_message}</div>}
                                    {status === 'failed' && (j.error_message || j.error_traceback) && (
                                        <details className={styles.details}>
                                            <summary>Failure details</summary>
                                            {j.error_message && <pre className={styles.pre}>{String(j.error_message)}</pre>}
                                            {j.error_traceback && <pre className={styles.pre}>{String(j.error_traceback)}</pre>}
                                        </details>
                                    )}
                                </div>
                            );
                        })
                    )}
                </div>
            )}
        </div>
    );
}
