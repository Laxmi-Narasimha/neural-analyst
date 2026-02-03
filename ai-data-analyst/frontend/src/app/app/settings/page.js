'use client';

import { useState } from 'react';
import { IconUser, IconMail, IconLock, IconKey, IconShield, IconArrowRight } from '@/components/icons';
import styles from './page.module.css';

const tabs = [
    { id: 'profile', label: 'Profile', Icon: IconUser },
    { id: 'api', label: 'API Keys', Icon: IconKey },
    { id: 'security', label: 'Security', Icon: IconShield },
];

export default function SettingsPage() {
    const [activeTab, setActiveTab] = useState('profile');

    return (
        <div className={styles.page}>
            <h1 className={styles.title}>Settings</h1>

            {/* Tabs */}
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

            {/* Content */}
            <div className={styles.content}>
                {activeTab === 'profile' && (
                    <div className={styles.section}>
                        <h2 className={styles.sectionTitle}>Profile Information</h2>
                        <form className={styles.form}>
                            <div className={styles.formGroup}>
                                <label className={styles.label}>Full Name</label>
                                <input type="text" defaultValue="John Doe" className={styles.input} />
                            </div>
                            <div className={styles.formGroup}>
                                <label className={styles.label}>Email Address</label>
                                <input type="email" defaultValue="john@company.com" className={styles.input} />
                            </div>
                            <div className={styles.formGroup}>
                                <label className={styles.label}>Company</label>
                                <input type="text" defaultValue="Acme Inc." className={styles.input} />
                            </div>
                            <button type="submit" className={styles.saveBtn}>
                                Save Changes
                            </button>
                        </form>
                    </div>
                )}

                {activeTab === 'api' && (
                    <div className={styles.section}>
                        <h2 className={styles.sectionTitle}>API Keys</h2>
                        <p className={styles.sectionDesc}>
                            Configure your AI provider keys for BYOK mode.
                        </p>
                        <div className={styles.apiCards}>
                            <div className={styles.apiCard}>
                                <div className={styles.apiHeader}>
                                    <span className={styles.apiName}>OpenAI</span>
                                    <span className={styles.apiStatus}>Configured</span>
                                </div>
                                <p className={styles.apiKey}>sk-proj-****...****7x2K</p>
                                <button className={styles.updateBtn}>Update Key</button>
                            </div>
                            <div className={styles.apiCard}>
                                <div className={styles.apiHeader}>
                                    <span className={styles.apiName}>Google Gemini</span>
                                    <span className={styles.apiStatusNot}>Not Configured</span>
                                </div>
                                <button className={styles.addBtn}>
                                    <IconKey size={16} />
                                    Add Key
                                </button>
                            </div>
                            <div className={styles.apiCard}>
                                <div className={styles.apiHeader}>
                                    <span className={styles.apiName}>Anthropic</span>
                                    <span className={styles.apiStatusNot}>Not Configured</span>
                                </div>
                                <button className={styles.addBtn}>
                                    <IconKey size={16} />
                                    Add Key
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'security' && (
                    <div className={styles.section}>
                        <h2 className={styles.sectionTitle}>Security Settings</h2>
                        <div className={styles.securityItems}>
                            <div className={styles.securityItem}>
                                <div className={styles.securityInfo}>
                                    <h3>Change Password</h3>
                                    <p>Update your account password</p>
                                </div>
                                <button className={styles.actionBtn}>
                                    <IconLock size={16} />
                                    Change
                                </button>
                            </div>
                            <div className={styles.securityItem}>
                                <div className={styles.securityInfo}>
                                    <h3>Two-Factor Authentication</h3>
                                    <p>Add an extra layer of security</p>
                                </div>
                                <button className={styles.actionBtn}>
                                    <IconShield size={16} />
                                    Enable
                                </button>
                            </div>
                            <div className={styles.securityItem}>
                                <div className={styles.securityInfo}>
                                    <h3>Active Sessions</h3>
                                    <p>Manage your logged-in devices</p>
                                </div>
                                <button className={styles.actionBtn}>
                                    <IconArrowRight size={16} />
                                    View
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
