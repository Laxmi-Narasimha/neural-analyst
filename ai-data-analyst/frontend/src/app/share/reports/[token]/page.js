'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import ReactMarkdown from 'react-markdown';
import api from '@/lib/api';
import styles from './page.module.css';

function formatDateTime(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString();
}

export default function SharedReportPage({ params }) {
    const token = params?.token ? String(params.token) : '';
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [report, setReport] = useState(null);

    useEffect(() => {
        if (!token) return;
        let cancelled = false;
        const run = async () => {
            try {
                setLoading(true);
                setError(null);
                const data = await api.getSharedReport(token);
                if (cancelled) return;
                setReport(data || null);
            } catch (e) {
                if (cancelled) return;
                setError(e?.message || 'Failed to load shared report');
                setReport(null);
            } finally {
                if (!cancelled) setLoading(false);
            }
        };
        run();
        return () => {
            cancelled = true;
        };
    }, [token]);

    const fmt = useMemo(() => String(report?.format || 'markdown').toLowerCase(), [report]);
    const content = useMemo(() => String(report?.content || ''), [report]);

    const download = async () => {
        try {
            const ext = fmt === 'html' ? 'html' : fmt === 'json' ? 'json' : 'md';
            const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${String(report?.artifact_id || 'report')}.${ext}`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            setTimeout(() => URL.revokeObjectURL(url), 10_000);
        } catch {
            // ignore
        }
    };

    return (
        <div className={styles.page}>
            <header className={styles.header}>
                <div className={styles.brand}>
                    <Link href="/" className={styles.logo}>Neural Analyst</Link>
                    <span className={styles.badge}>Shared Report</span>
                </div>
                <div className={styles.headerActions}>
                    <button className={styles.btn} onClick={download} type="button" disabled={!report || loading}>
                        Download
                    </button>
                    <Link className={styles.btnSecondary} href="/login">
                        Open App
                    </Link>
                </div>
            </header>

            {loading ? (
                <div className={styles.state}>Loading...</div>
            ) : error ? (
                <div className={styles.error}>
                    <div className={styles.errorTitle}>This share link is invalid or expired.</div>
                    <div className={styles.errorDetail}>{error}</div>
                </div>
            ) : (
                <div className={styles.card}>
                    <div className={styles.meta}>
                        <div className={styles.title}>{report?.name || 'Report'}</div>
                        <div className={styles.sub}>
                            <span><strong>Created:</strong> {formatDateTime(report?.created_at)}</span>
                            <span><strong>Format:</strong> {fmt}</span>
                        </div>
                    </div>

                    {fmt === 'html' ? (
                        <iframe
                            className={styles.iframe}
                            sandbox=""
                            srcDoc={content}
                            title="Shared report"
                        />
                    ) : (
                        <div className={styles.markdown}>
                            <ReactMarkdown>{content}</ReactMarkdown>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

