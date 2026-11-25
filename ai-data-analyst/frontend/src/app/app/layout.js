'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import {
    IconDashboard, IconDatabase, IconChart, IconSettings,
    IconPlus, IconSearch, IconUser, IconArrowRight, IconServer, IconSparkles, IconTime, IconBook
} from '@/components/icons';
import { clearTokens } from '@/lib/auth';
import api from '@/lib/api';
import styles from './layout.module.css';

const navItems = [
    { href: '/app/dashboard', label: 'Dashboard', Icon: IconDashboard },
    { href: '/app/datasets', label: 'Datasets', Icon: IconDatabase },
    { href: '/app/connections', label: 'Connections', Icon: IconServer },
    { href: '/app/analysis', label: 'Analysis', Icon: IconChart },
    { href: '/app/data-speaks', label: 'Data Speaks', Icon: IconSparkles },
    { href: '/app/jobs', label: 'Jobs', Icon: IconTime },
    { href: '/app/reports', label: 'Reports', Icon: IconBook },
    // Data Adequacy (/app/quality) is left out of the menu until its page is restyled without Tailwind.
    { href: '/app/settings', label: 'Settings', Icon: IconSettings },
];

export default function AppLayout({ children }) {
    const router = useRouter();
    const pathname = usePathname();
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [authReady, setAuthReady] = useState(false);
    const [userMenuOpen, setUserMenuOpen] = useState(false);
    const [currentUser, setCurrentUser] = useState(null);
    const [loggingOut, setLoggingOut] = useState(false);
    const [backendError, setBackendError] = useState(null);
    const [attempt, setAttempt] = useState(0);

    useEffect(() => {
        // The backend decides who is signed in: local mode needs no login, JWT mode needs a token.
        let cancelled = false;
        setBackendError(null);
        api.getCurrentUser()
            .then((res) => {
                if (cancelled) return;
                setCurrentUser(res?.data || res);
                setAuthReady(true);
            })
            .catch((err) => {
                if (cancelled) return;
                if (err?.status === 401 || err?.status === 403) {
                    router.replace('/login');
                } else {
                    setBackendError(err?.message || 'The backend did not answer');
                }
            });
        return () => { cancelled = true; };
    }, [router, attempt]);

    const handleLogout = async () => {
        setLoggingOut(true);
        try {
            await api.logout();
        } catch {
            clearTokens();
        } finally {
            setLoggingOut(false);
            setUserMenuOpen(false);
            router.replace('/login');
        }
    };

    if (backendError) {
        return (
            <div className={styles.layout} style={{ alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ maxWidth: 480, textAlign: 'center', color: 'var(--text-muted, #666)', lineHeight: 1.6 }}>
                    <p style={{ fontWeight: 600, color: 'var(--text-primary, #111)' }}>Cannot reach the Neural Analyst backend</p>
                    <p>Expected it at <code>{api.baseUrl}</code> ({backendError}). Start it with <code>python scripts/demo.py serve</code> or <code>make dev-analyst</code>, then retry.</p>
                    <button type="button" onClick={() => setAttempt((n) => n + 1)} style={{ marginTop: 12, padding: '8px 16px', borderRadius: 8, border: '1px solid #ccc', background: 'white', cursor: 'pointer' }}>
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    if (!authReady) {
        return (
            <div className={styles.layout} style={{ alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ color: 'var(--text-muted, #666)' }}>Loading...</div>
            </div>
        );
    }

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
                                    className={`${styles.navLink} ${pathname === item.href || pathname?.startsWith(`${item.href}/`) ? styles.active : ''}`}
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
                        <button
                            className={styles.userBtn}
                            type="button"
                            aria-label="User menu"
                            aria-expanded={userMenuOpen}
                            onClick={() => setUserMenuOpen((open) => !open)}
                        >
                            <IconUser size={18} />
                        </button>
                        {userMenuOpen && (
                            <div className={styles.userMenu} role="menu">
                                <div className={styles.userMenuHeader}>
                                    {currentUser?.full_name || currentUser?.email || 'Signed in'}
                                </div>
                                <Link
                                    href="/app/settings"
                                    className={styles.userMenuItem}
                                    role="menuitem"
                                    onClick={() => setUserMenuOpen(false)}
                                >
                                    Settings
                                </Link>
                                <button
                                    className={styles.userMenuItem}
                                    type="button"
                                    role="menuitem"
                                    onClick={handleLogout}
                                    disabled={loggingOut}
                                >
                                    {loggingOut ? 'Signing out…' : 'Sign out'}
                                </button>
                            </div>
                        )}
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
