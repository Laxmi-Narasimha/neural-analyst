'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import api from '@/lib/api';
import {
    IconPlus,
    IconSearch,
    IconChart,
    IconArrowRight,
    IconTime,
    IconLoader,
    IconX,
} from '@/components/icons';
import styles from './page.module.css';

const analysisTypes = [
    { id: 'all', label: 'All' },
    { id: 'eda', label: 'EDA' },
    { id: 'ml', label: 'Machine Learning' },
    { id: 'stats', label: 'Statistics' },
    { id: 'timeseries', label: 'Time Series' },
    { id: 'nlp', label: 'NLP' },
];

function toCategory(analysisType) {
    const t = String(analysisType || '').toLowerCase();
    if (t === 'eda') return 'eda';
    if (t === 'statistical') return 'stats';
    if (t === 'time_series') return 'timeseries';
    if (t === 'nlp') return 'nlp';
    if (t.startsWith('ml_') || t === 'deep_learning') return 'ml';
    return 'all';
}

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

export default function AnalysisPage() {
    const [activeType, setActiveType] = useState('all');
    const [searchQuery, setSearchQuery] = useState('');
    const [analyses, setAnalyses] = useState([]);
    const [datasetsById, setDatasetsById] = useState({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const load = async () => {
        try {
            setLoading(true);
            setError(null);

            // Avoid client-side waterfalls: fetch analyses + dataset list in parallel.
            const [analysesRes, datasetsRes] = await Promise.all([
                api.listAnalyses(1, 100),
                api.listDatasets(1, 200),
            ]);

            const items = analysesRes.items || [];
            const dsItems = datasetsRes.items || [];
            const map = {};
            for (const d of dsItems) {
                if (d?.id) map[String(d.id)] = d?.name || String(d.id);
            }

            setAnalyses(items);
            setDatasetsById(map);
        } catch (e) {
            setError(e?.message || 'Failed to load analyses');
            setAnalyses([]);
            setDatasetsById({});
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        load();
    }, []);

    const filteredAnalyses = useMemo(() => {
        const q = String(searchQuery || '').trim().toLowerCase();

        return (analyses || [])
            .filter((a) => {
                const cat = toCategory(a?.analysis_type);
                if (activeType !== 'all' && cat !== activeType) return false;

                if (!q) return true;
                const hay = [
                    a?.name,
                    a?.description,
                    a?.analysis_type,
                    a?.status,
                    a?.dataset_id,
                    datasetsById[String(a?.dataset_id)] || '',
                ]
                    .filter(Boolean)
                    .join(' ')
                    .toLowerCase();
                return hay.includes(q);
            })
            .sort((a, b) => new Date(b?.created_at || 0) - new Date(a?.created_at || 0));
    }, [analyses, datasetsById, activeType, searchQuery]);

    return (
        <div className={styles.page}>
            <div className={styles.header}>
                <div className={styles.headerLeft}>
                    <h1>Analysis History</h1>
                    <p>View and manage your analyses (EDA, stats, ML runs)</p>
                </div>
                <Link href="/app/analysis/new" className={styles.newBtn}>
                    <IconPlus size={18} />
                    New Analysis
                </Link>
            </div>

            <div className={styles.filters}>
                <div className={styles.tabs}>
                    {analysisTypes.map((type) => (
                        <button
                            key={type.id}
                            className={`${styles.tab} ${activeType === type.id ? styles.active : ''}`}
                            onClick={() => setActiveType(type.id)}
                            type="button"
                        >
                            {type.label}
                        </button>
                    ))}
                </div>
                <div className={styles.searchWrapper}>
                    <IconSearch size={18} className={styles.searchIcon} />
                    <input
                        type="text"
                        placeholder="Search analyses..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className={styles.searchInput}
                    />
                </div>
            </div>

            {error && (
                <motion.div className={styles.error} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
                    <IconX size={18} />
                    {error}
                    <button className={styles.retry} onClick={load} type="button">
                        Retry
                    </button>
                </motion.div>
            )}

            {loading ? (
                <div className={styles.loading}>
                    <IconLoader size={28} className={styles.spinning} />
                    <p>Loading analyses...</p>
                </div>
            ) : filteredAnalyses.length ? (
                <div className={styles.analysisList}>
                    {filteredAnalyses.map((analysis, index) => {
                        const status = String(analysis?.status || '').toLowerCase();
                        const datasetName =
                            datasetsById[String(analysis?.dataset_id)] || String(analysis?.dataset_id || '');
                        const created = formatRelative(analysis?.created_at);

                        return (
                            <motion.div
                                key={analysis.id}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: index * 0.03 }}
                            >
                                <Link href={`/app/analysis/${analysis.id}`} className={styles.analysisCard}>
                                    <div className={styles.analysisInfo}>
                                        <div className={styles.analysisIcon}>
                                            <IconChart size={20} />
                                        </div>
                                        <div className={styles.analysisDetails}>
                                            <h3>{analysis.name}</h3>
                                            <div className={styles.analysisMeta}>
                                                <span>{String(analysis?.analysis_type || '').toUpperCase()}</span>
                                                <span className={styles.metaSep}>-</span>
                                                <span title={String(analysis?.dataset_id || '')}>{datasetName}</span>
                                                {analysis?.status_message ? (
                                                    <>
                                                        <span className={styles.metaSep}>-</span>
                                                        <span className={styles.statusMessage}>
                                                            {analysis.status_message}
                                                        </span>
                                                    </>
                                                ) : null}
                                            </div>
                                        </div>
                                    </div>

                                    <div className={styles.right}>
                                        <div className={styles.analysisTime}>
                                            <IconTime size={14} />
                                            <span>{created}</span>
                                        </div>
                                        <span className={`${styles.analysisStatus} ${styles[status]}`}>
                                            {status}
                                            {status === 'running'
                                                ? ` ${Math.round((analysis?.progress || 0) * 100)}%`
                                                : ''}
                                        </span>
                                        <span className={styles.analysisAction} aria-label="Open analysis">
                                            <IconArrowRight size={18} />
                                        </span>
                                    </div>
                                </Link>
                            </motion.div>
                        );
                    })}
                </div>
            ) : (
                <div className={styles.empty}>
                    <div className={styles.emptyIcon}>
                        <IconChart size={32} />
                    </div>
                    <div className={styles.emptyTitle}>No analyses yet</div>
                    <div className={styles.emptyDesc}>
                        Create your first analysis to start generating evidence artifacts.
                    </div>
                </div>
            )}
        </div>
    );
}

