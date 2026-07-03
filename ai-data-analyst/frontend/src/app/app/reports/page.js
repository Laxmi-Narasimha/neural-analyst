'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import api from '@/lib/api';
import { IconDownload, IconLoader, IconTime, IconX } from '@/components/icons';
import styles from './page.module.css';

function formatDateTime(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString();
}

export default function ReportsPage() {
    const [reports, setReports] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [downloading, setDownloading] = useState(null);
    const [sharing, setSharing] = useState(null);
    const [shareUrl, setShareUrl] = useState(null);
    const [shareCopied, setShareCopied] = useState(false);

    const load = useCallback(async () => {
        try {
            setLoading(true);
            setError(null);
            const res = await api.listArtifacts(1, 100, { type: 'report' });
            setReports(res.items || []);
        } catch (e) {
            setError(e?.message || 'Failed to load reports');
            setReports([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        load();
    }, [load]);

    const sorted = useMemo(() => {
        const arr = Array.isArray(reports) ? reports.slice() : [];
        arr.sort((a, b) => String(b?.created_at || '').localeCompare(String(a?.created_at || '')));
        return arr;
    }, [reports]);

    const download = async (artifactId, formatHint = null) => {
        if (!artifactId) return;
        try {
            setDownloading(String(artifactId));
            const blob = await api.downloadArtifactData(artifactId);
            const url = URL.createObjectURL(blob);

            const fmt = String(formatHint || '').toLowerCase();
            const ext = fmt === 'markdown' || fmt === 'md' ? 'md' : fmt === 'html' ? 'html' : fmt === 'json' ? 'json' : fmt === 'text' ? 'txt' : '';
            const a = document.createElement('a');
            a.href = url;
            a.download = ext ? `${String(artifactId)}.${ext}` : String(artifactId);
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

    const share = async (artifactId) => {
        if (!artifactId) return;
        try {
            setSharing(String(artifactId));
            setError(null);
            setShareCopied(false);
            const res = await api.createReportShare(artifactId);
            const path = res?.share_path ? String(res.share_path) : '';
            const url = path ? `${window.location.origin}${path}` : '';
            if (!url) throw new Error('Share link missing');
            setShareUrl(url);
            try {
                if (navigator?.clipboard?.writeText) {
                    await navigator.clipboard.writeText(url);
                    setShareCopied(true);
                }
            } catch {
                // clipboard may be unavailable in some contexts
                setShareCopied(false);
            }
        } catch (e) {
            setError(e?.message || 'Share failed');
        } finally {
            setSharing(null);
        }
    };

    return (
        <div className={styles.page}>
            <div className={styles.header}>
                <div className={styles.titleRow}>
                    <div className={styles.titleIcon}><IconTime size={18} /></div>
                    <div>
                        <h1 className={styles.title}>Reports</h1>
                        <p className={styles.subtitle}>Exported grounded reports (markdown/html) generated from artifacts.</p>
                    </div>
                </div>

                <button className={styles.refreshBtn} onClick={load} disabled={loading} type="button">
                    {loading ? <IconLoader size={16} className={styles.spinning} /> : 'Refresh'}
                </button>
            </div>

            {error && (
                <motion.div className={styles.error} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
                    <IconX size={18} />
                    {error}
                    <button className={styles.dismiss} onClick={() => setError(null)} type="button">Dismiss</button>
                </motion.div>
            )}

            {shareUrl ? (
                <motion.div className={styles.shareBanner} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
                    <div className={styles.shareText}>
                        <strong>Share link</strong>
                        <span className={styles.shareMeta}>{shareCopied ? 'Copied to clipboard' : 'Copy it manually'}</span>
                        <code className={styles.shareUrl}>{shareUrl}</code>
                    </div>
                    <div className={styles.shareActions}>
                        <a className={styles.shareOpen} href={shareUrl} target="_blank" rel="noreferrer">Open</a>
                        <button className={styles.shareClose} onClick={() => setShareUrl(null)} type="button">Close</button>
                    </div>
                </motion.div>
            ) : null}

            {loading ? (
                <div className={styles.loading}>
                    <IconLoader size={32} className={styles.spinning} />
                    <p>Loading reports...</p>
                </div>
            ) : (
                <div className={styles.list}>
                    {sorted.length === 0 ? (
                        <div className={styles.empty}>No reports yet. Export a report from an Analysis or Data Speaks run.</div>
                    ) : (
                        sorted.map((r) => {
                            const preview = r?.preview && typeof r.preview === 'object' ? r.preview : {};
                            const fmt = preview?.format ? String(preview.format) : '';
                            const snippet = preview?.preview_text ? String(preview.preview_text) : '';
                            return (
                                <div key={r.id} className={styles.card}>
                                    <div className={styles.cardTop}>
                                        <div className={styles.mainInfo}>
                                            <div className={styles.badges}>
                                                <span className={styles.badge}>report</span>
                                                {fmt ? <span className={styles.badgeMuted}>{fmt}</span> : null}
                                            </div>
                                            <div className={styles.reportName}>{r.name || r.id}</div>
                                            <div className={styles.meta}>
                                                <div><strong>Created:</strong> {formatDateTime(r.created_at)}</div>
                                                {r.dataset_id ? <div><strong>Dataset:</strong> {r.dataset_id}</div> : null}
                                                {r.operator_name ? <div><strong>Operator:</strong> {r.operator_name}</div> : null}
                                            </div>
                                        </div>

                                        <div className={styles.cardActions}>
                                            <button
                                                className={styles.shareBtn}
                                                onClick={() => share(r.id)}
                                                type="button"
                                                disabled={sharing === String(r.id)}
                                                title="Create a share link"
                                            >
                                                {sharing === String(r.id) ? <IconLoader size={16} className={styles.spinning} /> : 'Share'}
                                            </button>
                                            <button
                                                className={styles.downloadBtn}
                                                onClick={() => download(r.id, fmt)}
                                                type="button"
                                                disabled={downloading === String(r.id)}
                                                title="Download report"
                                            >
                                                {downloading === String(r.id) ? (
                                                    <IconLoader size={16} className={styles.spinning} />
                                                ) : (
                                                    <IconDownload size={16} />
                                                )}
                                                Download
                                            </button>
                                        </div>
                                    </div>

                                    {snippet ? (
                                        <pre className={styles.snippet}>{snippet}</pre>
                                    ) : (
                                        <div className={styles.muted}>No preview available.</div>
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
