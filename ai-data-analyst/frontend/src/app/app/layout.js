'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    IconDashboard, IconDatabase, IconChart, IconSettings,
    IconPlus, IconSearch, IconUser, IconArrowRight, IconServer, IconShield
} from '@/components/icons';
import styles from './layout.module.css';

const navItems = [
    { href: '/app/dashboard', label: 'Dashboard', Icon: IconDashboard },
    { href: '/app/datasets', label: 'Datasets', Icon: IconDatabase },
    { href: '/app/connections', label: 'Connections', Icon: IconServer },
    { href: '/app/analysis', label: 'Analysis', Icon: IconChart },
    { href: '/app/quality', label: 'Data Quality', Icon: IconShield },
    { href: '/app/settings', label: 'Settings', Icon: IconSettings },
];

export default function AppLayout({ children }) {
    const pathname = usePathname();
    const [sidebarOpen, setSidebarOpen] = useState(true);

    return (
        <div className={styles.layout}>
            {/* Sidebar */}
            <aside className={`${styles.sidebar} ${sidebarOpen ? '' : styles.collapsed}`}>
                <div className={styles.sidebarHeader}>
                    <Link href="/" className={styles.logo}>
                        <div className={styles.logoIcon}>
                            <svg viewBox="0 0 32 32" fill="none">
                                <circle cx="16" cy="16" r="14" stroke="url(#appGradient)" strokeWidth="2" />
                                <path d="M10 16L14 20L22 12" stroke="url(#appGradient)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                <defs>
                                    <linearGradient id="appGradient" x1="0" y1="0" x2="32" y2="32">
                                        <stop stopColor="#06b6d4" />
                                        <stop offset="1" stopColor="#8b5cf6" />
                                    </linearGradient>
                                </defs>
                            </svg>
                        </div>
                        {sidebarOpen && <span className={styles.logoText}>Neural<span>Analyst</span></span>}
                    </Link>
                </div>

                <nav className={styles.nav}>
                    <Link href="/app/analysis/new" className={styles.newAnalysis}>
                        <IconPlus size={18} />
                        {sidebarOpen && <span>New Analysis</span>}
                    </Link>

                    <ul className={styles.navList}>
                        {navItems.map((item) => (
                            <li key={item.href}>
                                <Link
                                    href={item.href}
                                    className={`${styles.navLink} ${pathname === item.href ? styles.active : ''}`}
                                >
                                    <item.Icon size={20} />
                                    {sidebarOpen && <span>{item.label}</span>}
                                </Link>
                            </li>
                        ))}
                    </ul>
                </nav>

                <div className={styles.sidebarFooter}>
                    <button
                        className={styles.toggleBtn}
                        onClick={() => setSidebarOpen(!sidebarOpen)}
                        aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
                    >
                        <IconArrowRight size={16} className={sidebarOpen ? styles.rotated : ''} />
                    </button>
                </div>
            </aside>

            {/* Main Content */}
            <div className={styles.main}>
                {/* Top Bar */}
                <header className={styles.topbar}>
                    <div className={styles.searchWrapper}>
                        <IconSearch size={18} className={styles.searchIcon} />
                        <input
                            type="text"
                            placeholder="Search datasets, analyses..."
                            className={styles.searchInput}
                        />
                    </div>
                    <div className={styles.userSection}>
                        <button className={styles.userBtn}>
                            <IconUser size={18} />
                        </button>
                    </div>
                </header>

                {/* Page Content */}
                <main className={styles.content}>
                    {children}
                </main>
            </div>
        </div>
    );
}
