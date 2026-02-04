'use client';

import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import api from '@/lib/api';
import { IconDatabase, IconDownload, IconLoader, IconSparkles, IconX } from '@/components/icons';
import styles from './page.module.css';

function safeJson(value) {
    try {
        return JSON.stringify(value, null, 2);
    } catch {
        return String(value);
    }
}

export default function DataSpeaksPage() {
    const [datasets, setDatasets] = useState([]);
    const [selectedDatasetId, setSelectedDatasetId] = useState('');
    const [loadingDatasets, setLoadingDatasets] = useState(true);
    const [running, setRunning] = useState(false);
    const [error, setError] = useState(null);
    const [runResult, setRunResult] = useState(null);

    useEffect(() => {
        const load = async () => {
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
        };

        load();
    }, []);

    useEffect(() => {
        if (!selectedDatasetId && datasets?.[0]?.id) {
            setSelectedDatasetId(datasets[0].id);
        }
    }, [datasets, selectedDatasetId]);

    const selectedDataset = useMemo(
        () => datasets.find((d) => String(d.id) === String(selectedDatasetId)) || null,
        [datasets, selectedDatasetId]
    );

    const selectedStatus = String(selectedDataset?.status || '').toLowerCase();
    const canRun = Boolean(selectedDatasetId && selectedStatus === 'ready');

    const run = async () => {
        if (!selectedDatasetId) return;
        try {
            setRunning(true);
            setError(null);
            setRunResult(null);
            const result = await api.runDataSpeaks(selectedDatasetId);
            setRunResult(result);
        } catch (e) {
            setError(e?.message || 'Data Speaks run failed');
        } finally {
            setRunning(false);
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

    return (
        <div className={styles.page}>
            <div className={styles.header}>
                <div className={styles.titleRow}>
                    <div className={styles.titleIcon}>
                        <IconSparkles size={18} />
                    </div>
                    <div>
                        <h1 className={styles.title}>Data Speaks</h1>
                        <p className={styles.subtitle}>Run safe, computed operators and browse evidence artifacts.</p>
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

                    <button className={styles.runBtn} onClick={run} disabled={!canRun || running || loadingDatasets}>
                        {running ? <IconLoader size={16} className={styles.spinning} /> : <IconSparkles size={16} />}
                        {running ? 'Running...' : 'Make the data speak'}
                    </button>
                </div>
            </div>

            {error && (
                <motion.div
                    className={styles.error}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <IconX size={18} />
                    {error}
                    <button className={styles.dismiss} onClick={() => setError(null)}>Dismiss</button>
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

            {!runResult && !running && (
                <div className={styles.empty}>
                    <p>Select a dataset and run Data Speaks to generate evidence artifacts.</p>
                </div>
            )}

            {runResult?.steps?.length ? (
                <div className={styles.steps}>
                    {runResult.steps.map((step, idx) => (
                        <div key={`${step.operator}-${idx}`} className={styles.stepCard}>
                            <div className={styles.stepHeader}>
                                <div className={styles.stepTitle}>
                                    <span className={styles.stepIndex}>{idx + 1}</span>
                                    <span>{step.operator}</span>
                                </div>
                            </div>

                            {step.summary && (
                                <pre className={styles.pre}>{safeJson(step.summary)}</pre>
                            )}

                            {step.artifacts?.length ? (
                                <div className={styles.artifacts}>
                                    {step.artifacts.map((a) => (
                                        <div key={a.artifact_id} className={styles.artifact}>
                                            <div className={styles.artifactHeader}>
                                                <div className={styles.artifactTitle}>
                                                    <span className={styles.badge}>{a.artifact_type}</span>
                                                    <span className={styles.artifactName}>{a.name}</span>
                                                </div>
                                                <button
                                                    className={styles.downloadBtn}
                                                    onClick={() => downloadArtifact(a.artifact_id)}
                                                    title="Download artifact data"
                                                >
                                                    <IconDownload size={16} />
                                                </button>
                                            </div>

                                            {a.preview && (
                                                <pre className={styles.preSmall}>{safeJson(a.preview)}</pre>
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
            ) : null}
        </div>
    );
}
