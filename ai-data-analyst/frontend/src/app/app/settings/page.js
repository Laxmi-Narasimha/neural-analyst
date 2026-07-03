'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { IconUser, IconKey, IconShield, IconArrowRight } from '@/components/icons';
import api from '@/lib/api';
import styles from './page.module.css';

const tabs = [
    { id: 'profile', label: 'Profile', Icon: IconUser },
    { id: 'billing', label: 'Billing', Icon: IconKey },
    { id: 'security', label: 'Security', Icon: IconShield },
];

export default function SettingsPage() {
    const searchParams = useSearchParams();
    const [activeTab, setActiveTab] = useState('profile');
    const [loading, setLoading] = useState(true);
    const [billingLoading, setBillingLoading] = useState(false);
    const [profile, setProfile] = useState(null);
    const [subscription, setSubscription] = useState(null);
    const [error, setError] = useState(null);
    const [billingNotice, setBillingNotice] = useState(null);

    useEffect(() => {
        const billing = searchParams.get('billing');
        if (billing === 'success') {
            setBillingNotice('Subscription updated successfully.');
            setActiveTab('billing');
        }
    }, [searchParams]);

    useEffect(() => {
        const load = async () => {
            try {
                setLoading(true);
                const [me, sub] = await Promise.all([
                    api.getCurrentUser(),
                    api.getSubscriptionStatus(),
                ]);
                setProfile(me);
                setSubscription(sub);
            } catch (e) {
                setError(e?.message || 'Failed to load settings');
            } finally {
                setLoading(false);
            }
        };
        load();
    }, []);

    const handleUpgrade = async (plan) => {
        try {
            setBillingLoading(true);
            const session = await api.createCheckoutSession(plan);
            if (session?.checkout_url) {
                window.location.href = session.checkout_url;
                return;
            }
            setError('Checkout is not configured on this deployment.');
        } catch (e) {
            setError(e?.message || 'Could not start checkout');
        } finally {
            setBillingLoading(false);
        }
    };

    const handlePortal = async () => {
        try {
            setBillingLoading(true);
            const portal = await api.openBillingPortal();
            if (portal?.portal_url) {
                window.location.href = portal.portal_url;
            }
        } catch (e) {
            setError(e?.message || 'Billing portal unavailable');
        } finally {
            setBillingLoading(false);
        }
    };

    if (loading) {
        return <div className={styles.page}><p>Loading settings...</p></div>;
    }

    const plan = String(subscription?.plan || 'free');
    const usage = subscription?.usage || {};
    const limits = subscription?.limits || {};
    const enforcement = Boolean(subscription?.enforcement_enabled);

    return (
        <div className={styles.page}>
            <h1 className={styles.title}>Settings</h1>
            {error && <p className={styles.errorText}>{error}</p>}
            {billingNotice && <p className={styles.noticeText}>{billingNotice}</p>}

            <div className={styles.tabs}>
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        className={`${styles.tab} ${activeTab === tab.id ? styles.active : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        <tab.Icon size={18} />
                        {tab.label}
                    </button>
                ))}
            </div>

            <div className={styles.content}>
                {activeTab === 'profile' && (
                    <div className={styles.section}>
                        <h2 className={styles.sectionTitle}>Profile</h2>
                        <div className={styles.form}>
                            <div className={styles.formGroup}>
                                <label className={styles.label}>Email</label>
                                <input type="email" value={profile?.email || ''} className={styles.input} readOnly />
                            </div>
                            <div className={styles.formGroup}>
                                <label className={styles.label}>Role</label>
                                <input type="text" value={profile?.role || ''} className={styles.input} readOnly />
                            </div>
                            <div className={styles.formGroup}>
                                <label className={styles.label}>Deployment</label>
                                <input
                                    type="text"
                                    value={subscription?.deployment_mode || 'self_host'}
                                    className={styles.input}
                                    readOnly
                                />
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'billing' && (
                    <div className={styles.section}>
                        <h2 className={styles.sectionTitle}>Plan & Usage</h2>
                        <p className={styles.sectionDesc}>
                            {enforcement
                                ? 'Hosted SaaS mode: one free Talk-to-Your-Data preview, then upgrade or self-host.'
                                : 'Self-host mode: no usage limits. You control keys and compute.'}
                        </p>

                        <div className={styles.billingCard}>
                            <div className={styles.billingRow}>
                                <span>Current plan</span>
                                <strong>{plan.toUpperCase()}</strong>
                            </div>
                            <div className={styles.billingRow}>
                                <span>Preview sessions used</span>
                                <strong>
                                    {usage.talk_preview_used || 0}
                                    {limits.talk_preview_sessions != null ? ` / ${limits.talk_preview_sessions}` : ''}
                                </strong>
                            </div>
                            <div className={styles.billingRow}>
                                <span>Monthly queries</span>
                                <strong>
                                    {usage.monthly_queries || 0}
                                    {limits.monthly_queries != null ? ` / ${limits.monthly_queries}` : ' / unlimited'}
                                </strong>
                            </div>
                        </div>

                        <div className={styles.billingActions}>
                            {plan === 'free' && enforcement && (
                                <>
                                    <button
                                        type="button"
                                        className={styles.saveBtn}
                                        disabled={billingLoading}
                                        onClick={() => handleUpgrade('pro')}
                                    >
                                        Upgrade to Pro
                                    </button>
                                    <button
                                        type="button"
                                        className={styles.actionBtn}
                                        disabled={billingLoading}
                                        onClick={() => handleUpgrade('enterprise')}
                                    >
                                        Upgrade to Enterprise
                                    </button>
                                </>
                            )}
                            {plan !== 'free' && subscription?.stripe_customer_id && (
                                <button type="button" className={styles.actionBtn} disabled={billingLoading} onClick={handlePortal}>
                                    Manage subscription
                                </button>
                            )}
                            <Link href="/pricing" className={styles.linkBtn}>View pricing</Link>
                            <a
                                href={subscription?.self_host_url || 'https://github.com/Laxmi-Narasimha/neural-analyst'}
                                className={styles.linkBtn}
                                target="_blank"
                                rel="noreferrer"
                            >
                                Self-host guide
                            </a>
                        </div>
                    </div>
                )}

                {activeTab === 'security' && (
                    <div className={styles.section}>
                        <h2 className={styles.sectionTitle}>Security</h2>
                        <div className={styles.securityItems}>
                            <div className={styles.securityItem}>
                                <div className={styles.securityInfo}>
                                    <h3>Self-host for full control</h3>
                                    <p>Run locally or on your cloud with your own API keys and data residency.</p>
                                </div>
                                <Link href="https://github.com/Laxmi-Narasimha/neural-analyst" className={styles.actionBtn}>
                                    <IconArrowRight size={16} />
                                    GitHub
                                </Link>
                            </div>
                            <div className={styles.securityItem}>
                                <div className={styles.securityInfo}>
                                    <h3>Evidence-first analysis</h3>
                                    <p>Dataset numbers always come from computed artifacts, not LLM guesses.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}