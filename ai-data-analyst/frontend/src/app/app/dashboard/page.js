'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { IconPlus, IconUpload, IconArrowRight, IconChart, IconDatabase, IconTrend, IconTime } from '@/components/icons';
import styles from './page.module.css';

const recentAnalyses = [
    { id: 1, name: 'Q4 Sales Analysis', type: 'EDA', date: '2 hours ago', status: 'complete' },
    { id: 2, name: 'Customer Churn Model', type: 'ML', date: '1 day ago', status: 'complete' },
    { id: 3, name: 'Marketing A/B Test', type: 'Stats', date: '3 days ago', status: 'complete' },
];

const recentDatasets = [
    { id: 1, name: 'sales_2024.csv', rows: '45,234', cols: 12, size: '4.2 MB' },
    { id: 2, name: 'customers.parquet', rows: '128,500', cols: 28, size: '15.8 MB' },
    { id: 3, name: 'transactions.xlsx', rows: '8,921', cols: 8, size: '1.1 MB' },
];

export default function DashboardPage() {
    return (
        <div className={styles.dashboard}>
            {/* Header */}
            <div className={styles.header}>
                <div>
                    <h1 className={styles.title}>Welcome back</h1>
                    <p className={styles.subtitle}>Your AI analyst is ready to work</p>
                </div>
                <Link href="/app/analysis/new" className={styles.newBtn}>
                    <IconPlus size={18} />
                    New Analysis
                </Link>
            </div>

            {/* Stats */}
            <div className={styles.statsGrid}>
                <motion.div
                    className={styles.statCard}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                >
                    <div className={styles.statIcon}>
                        <IconChart size={20} />
                    </div>
                    <div className={styles.statContent}>
                        <span className={styles.statValue}>24</span>
                        <span className={styles.statLabel}>Analyses This Month</span>
                    </div>
                </motion.div>
                <motion.div
                    className={styles.statCard}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                >
                    <div className={styles.statIcon}>
                        <IconDatabase size={20} />
                    </div>
                    <div className={styles.statContent}>
                        <span className={styles.statValue}>8</span>
                        <span className={styles.statLabel}>Active Datasets</span>
                    </div>
                </motion.div>
                <motion.div
                    className={styles.statCard}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                >
                    <div className={styles.statIcon}>
                        <IconTrend size={20} />
                    </div>
                    <div className={styles.statContent}>
                        <span className={styles.statValue}>2.1M</span>
                        <span className={styles.statLabel}>Rows Processed</span>
                    </div>
                </motion.div>
                <motion.div
                    className={styles.statCard}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                >
                    <div className={styles.statIcon}>
                        <IconTime size={20} />
                    </div>
                    <div className={styles.statContent}>
                        <span className={styles.statValue}>4.2h</span>
                        <span className={styles.statLabel}>Time Saved</span>
                    </div>
                </motion.div>
            </div>

            {/* Quick Actions */}
            <div className={styles.quickActions}>
                <h2 className={styles.sectionTitle}>Quick Actions</h2>
                <div className={styles.actionGrid}>
                    <Link href="/app/analysis/new" className={styles.actionCard}>
                        <div className={styles.actionIcon}>
                            <IconChart size={24} />
                        </div>
                        <div className={styles.actionContent}>
                            <h3>Start Analysis</h3>
                            <p>Ask questions about your data in natural language</p>
                        </div>
                        <IconArrowRight size={18} className={styles.actionArrow} />
                    </Link>
                    <Link href="/app/datasets/upload" className={styles.actionCard}>
                        <div className={styles.actionIcon}>
                            <IconUpload size={24} />
                        </div>
                        <div className={styles.actionContent}>
                            <h3>Upload Dataset</h3>
                            <p>CSV, Excel, JSON, Parquet supported</p>
                        </div>
                        <IconArrowRight size={18} className={styles.actionArrow} />
                    </Link>
                </div>
            </div>

            {/* Recent Activity */}
            <div className={styles.recentGrid}>
                {/* Recent Analyses */}
                <div className={styles.recentSection}>
                    <div className={styles.sectionHeader}>
                        <h2 className={styles.sectionTitle}>Recent Analyses</h2>
                        <Link href="/app/analysis" className={styles.viewAll}>View All</Link>
                    </div>
                    <div className={styles.recentList}>
                        {recentAnalyses.map((analysis) => (
                            <Link key={analysis.id} href={`/app/analysis/${analysis.id}`} className={styles.recentItem}>
                                <div className={styles.itemInfo}>
                                    <span className={styles.itemName}>{analysis.name}</span>
                                    <span className={styles.itemMeta}>{analysis.type} • {analysis.date}</span>
                                </div>
                                <span className={`${styles.status} ${styles[analysis.status]}`}>
                                    {analysis.status}
                                </span>
                            </Link>
                        ))}
                    </div>
                </div>

                {/* Recent Datasets */}
                <div className={styles.recentSection}>
                    <div className={styles.sectionHeader}>
                        <h2 className={styles.sectionTitle}>Recent Datasets</h2>
                        <Link href="/app/datasets" className={styles.viewAll}>View All</Link>
                    </div>
                    <div className={styles.recentList}>
                        {recentDatasets.map((dataset) => (
                            <Link key={dataset.id} href={`/app/datasets/${dataset.id}`} className={styles.recentItem}>
                                <div className={styles.itemInfo}>
                                    <span className={styles.itemName}>{dataset.name}</span>
                                    <span className={styles.itemMeta}>{dataset.rows} rows • {dataset.cols} cols</span>
                                </div>
                                <span className={styles.datasetSize}>{dataset.size}</span>
                            </Link>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
