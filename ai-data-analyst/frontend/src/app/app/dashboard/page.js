'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import api from '@/lib/api';
import { IconPlus, IconArrowRight, IconChart, IconDatabase, IconTrend, IconTime, IconX, IconLoader } from '@/components/icons';
import styles from './page.module.css';

function formatRelative(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    const ms = Date.now() - d.getTime();
    if (!Number.isFinite(ms)) return '';

    const minutes = Math.floor(ms / 60_000);
    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 30) return `${days}d ago`;
    return d.toLocaleDateString();
}

function formatCompactInt(n) {
    const v = Number(n || 0);
    if (!Number.isFinite(v)) return '0';
    if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1).replace('.0', '')}M`;
    if (v >= 1_000) return `${(v / 1_000).toFixed(1).replace('.0', '')}K`;
    return String(Math.trunc(v));
}

function formatHours(seconds) {
    const s = Number(seconds || 0);
    if (!Number.isFinite(s)) return '0h';
    return `${(s / 3600).toFixed(1).replace('.0', '')}h`;
}

export default function DashboardPage() {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const [summary, setSummary] = useState(null);
    const [recentAnalyses, setRecentAnalyses] = useState([]);
    const [recentDatasets, setRecentDatasets] = useState([]);

    const load = async () => {
        try {
            setLoading(true);
            setError(null);

            const [summaryRes, analysesRes, datasetsRes] = await Promise.all([
                api.getDashboardSummary(),
                api.listAnalyses(1, 5),
                api.listDatasets(1, 5),
            ]);

            setSummary(summaryRes || null);
            setRecentAnalyses(analysesRes.items || []);
            setRecentDatasets(datasetsRes.items || []);
        } catch (e) {
            setError(e?.message || 'Failed to load dashboard');
            setSummary(null);
            setRecentAnalyses([]);
            setRecentDatasets([]);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        load();
    }, []);

    const stats = useMemo(() => {
        const analysesThisMonth = summary?.analyses_this_month ?? 0;
        const datasetsReady = summary?.datasets_ready ?? 0;
        const rowsProcessed = summary?.rows_processed ?? 0;
        const computeSeconds = summary?.compute_seconds ?? 0;

        return [
            { key: 'analyses', label: 'Analyses This Month', value: formatCompactInt(analysesThisMonth), Icon: IconChart },
            { key: 'datasets', label: 'Ready Datasets', value: formatCompactInt(datasetsReady), Icon: IconDatabase },
            { key: 'rows', label: 'Rows Processed', value: formatCompactInt(rowsProcessed), Icon: IconTrend },
            { key: 'compute', label: 'Compute Time', value: formatHours(computeSeconds), Icon: IconTime },
        ];
    }, [summary]);

    return (
        <div className={styles.main}>
            <div className={styles.header}>
                <div className={styles.greeting}>
                    <h1>Welcome back</h1>
                    <p>Your AI analyst is ready to work (grounded by compute + artifacts)</p>
                </div>
                <div className={styles.actions}>
                    <Link href="/app/analysis/new" className={styles.primaryBtn}>
                        <IconPlus size={18} />
                        New Analysis
                    </Link>
                    <Link href="/app/datasets" className={styles.secondaryBtn}>
                        <IconDatabase size={18} />
                        Datasets
                    </Link>
                </div>
            </div>

            {error && (
                <motion.div className={styles.error} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
                    <IconX size={18} />
                    {error}
                    <button className={styles.retry} onClick={load} type="button">Retry</button>
                </motion.div>
            )}

            <div className={styles.stats}>
                {stats.map((s, idx) => (
                    <motion.div
                        key={s.key}
                        className={styles.statCard}
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.06 }}
                    >
                        <div className={styles.statIcon}>
                            {loading ? <IconLoader size={20} className={styles.spinning} /> : <s.Icon size={20} />}
                        </div>
                        <div className={styles.statLabel}>{s.label}</div>
                        <div className={styles.statValue}>{loading ? '—' : s.value}</div>
                    </motion.div>
                ))}
            </div>

            <div className={styles.sections}>
                <div className={styles.section}>
                    <div className={styles.sectionHeader}>
                        <div className={styles.sectionTitle}>Recent Analyses</div>
                        <Link href="/app/analysis" className={styles.sectionLink}>View all</Link>
                    </div>
                    {loading ? (
                        <div className={styles.empty}>
                            <div className={styles.emptyIcon}><IconLoader size={26} className={styles.spinning} /></div>
                            <p>Loading analyses…</p>
                        </div>
                    ) : recentAnalyses.length ? (
                        <div className={styles.itemList}>
                            {recentAnalyses.map((a) => (
                                <Link key={a.id} href={`/app/analysis/${a.id}`} className={styles.item}>
                                    <div className={styles.itemInfo}>
                                        <div className={styles.itemName}>{a.name}</div>
                                        <div className={styles.itemMeta}>
                                            {String(a.analysis_type || '').toUpperCase()} • {formatRelative(a.created_at)} • {String(a.status || '').toLowerCase()}
                                        </div>
                                    </div>
                                    <IconArrowRight size={18} className={styles.itemArrow} />
                                </Link>
                            ))}
                        </div>
                    ) : (
                        <div className={styles.empty}>
                            <div className={styles.emptyIcon}><IconChart size={28} /></div>
                            <p>No analyses yet.</p>
                        </div>
                    )}
                </div>

                <div className={styles.section}>
                    <div className={styles.sectionHeader}>
                        <div className={styles.sectionTitle}>Recent Datasets</div>
                        <Link href="/app/datasets" className={styles.sectionLink}>View all</Link>
                    </div>
                    {loading ? (
                        <div className={styles.empty}>
                            <div className={styles.emptyIcon}><IconLoader size={26} className={styles.spinning} /></div>
                            <p>Loading datasets…</p>
                        </div>
                    ) : recentDatasets.length ? (
                        <div className={styles.itemList}>
                            {recentDatasets.map((d) => (
                                <Link key={d.id} href={`/app/analysis/new?dataset=${d.id}`} className={styles.item}>
                                    <div className={styles.itemInfo}>
                                        <div className={styles.itemName}>{d.name}</div>
                                        <div className={styles.itemMeta}>
                                            {(d.row_count || 0).toLocaleString()} rows • {d.column_count || 0} cols • {String(d.status || '').toLowerCase()}
                                        </div>
                                    </div>
                                    <IconArrowRight size={18} className={styles.itemArrow} />
                                </Link>
                            ))}
                        </div>
                    ) : (
                        <div className={styles.empty}>
                            <div className={styles.emptyIcon}><IconDatabase size={28} /></div>
                            <p>No datasets yet.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

