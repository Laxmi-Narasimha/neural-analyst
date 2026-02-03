'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { IconDatabase, IconPlus, IconTrash, IconCheck, IconX, IconLoader, IconServer } from '@/components/icons';
import api from '@/lib/api';
import styles from './page.module.css';

const CONNECTION_TYPES = [
    { value: 'postgresql', label: 'PostgreSQL', port: 5432 },
    { value: 'mysql', label: 'MySQL', port: 3306 },
    { value: 'sqlite', label: 'SQLite', port: null },
    { value: 'mongodb', label: 'MongoDB', port: 27017 },
    { value: 'bigquery', label: 'BigQuery', port: null },
    { value: 'snowflake', label: 'Snowflake', port: 443 },
];

export default function ConnectionsPage() {
    const [connections, setConnections] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showForm, setShowForm] = useState(false);
    const [testing, setTesting] = useState(null);
    const [testResult, setTestResult] = useState({});
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState(null);

    const [formData, setFormData] = useState({
        name: '',
        connector_type: 'postgresql',
        host: 'localhost',
        port: 5432,
        database: '',
        username: '',
        password: '',
        ssl: false,
    });

    useEffect(() => {
        loadConnections();
    }, []);

    const loadConnections = async () => {
        try {
            setLoading(true);
            const response = await api.listConnections();
            setConnections(response.items || []);
        } catch (error) {
            console.error('Failed to load connections:', error);
            // Use mock data
            setConnections([
                { id: '1', name: 'Production DB', connector_type: 'postgresql', host: 'db.example.com', database: 'analytics', status: 'active' },
                { id: '2', name: 'Data Warehouse', connector_type: 'snowflake', host: 'account.snowflakecomputing.com', database: 'warehouse', status: 'active' },
            ]);
        } finally {
            setLoading(false);
        }
    };

    const handleTypeChange = (type) => {
        const typeConfig = CONNECTION_TYPES.find(t => t.value === type);
        setFormData(prev => ({
            ...prev,
            connector_type: type,
            port: typeConfig?.port || prev.port,
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!formData.name || !formData.database) {
            setError('Name and database are required');
            return;
        }

        try {
            setSaving(true);
            setError(null);
            await api.createConnection(formData);
            await loadConnections();
            setShowForm(false);
            resetForm();
        } catch (error) {
            setError(error.message || 'Failed to create connection');
        } finally {
            setSaving(false);
        }
    };

    const resetForm = () => {
        setFormData({
            name: '',
            connector_type: 'postgresql',
            host: 'localhost',
            port: 5432,
            database: '',
            username: '',
            password: '',
            ssl: false,
        });
    };

    const testConnection = async (connectionId) => {
        try {
            setTesting(connectionId);
            setTestResult(prev => ({ ...prev, [connectionId]: null }));

            const result = await api.testConnection(connectionId);
            setTestResult(prev => ({ ...prev, [connectionId]: result.success ? 'success' : 'failed' }));
        } catch (error) {
            setTestResult(prev => ({ ...prev, [connectionId]: 'failed' }));
        } finally {
            setTesting(null);
        }
    };

    const deleteConnection = async (connectionId) => {
        if (!confirm('Are you sure you want to delete this connection?')) return;

        try {
            await api.deleteConnection(connectionId);
            await loadConnections();
        } catch (error) {
            setError(error.message || 'Failed to delete connection');
        }
    };

    return (
        <div className={styles.page}>
            {/* Header */}
            <div className={styles.header}>
                <div>
                    <h1 className={styles.title}>Database Connections</h1>
                    <p className={styles.subtitle}>Connect to external databases and data warehouses</p>
                </div>
                <button
                    className={styles.addBtn}
                    onClick={() => setShowForm(true)}
                >
                    <IconPlus size={18} />
                    Add Connection
                </button>
            </div>

            {/* Error Message */}
            {error && (
                <motion.div
                    className={styles.errorMessage}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                >
                    <IconX size={18} />
                    {error}
                    <button onClick={() => setError(null)}>Dismiss</button>
                </motion.div>
            )}

            {/* New Connection Form */}
            {showForm && (
                <motion.div
                    className={styles.formCard}
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <div className={styles.formHeader}>
                        <h2>New Connection</h2>
                        <button onClick={() => { setShowForm(false); resetForm(); }}>
                            <IconX size={20} />
                        </button>
                    </div>

                    <form onSubmit={handleSubmit} className={styles.form}>
                        <div className={styles.formRow}>
                            <div className={styles.formGroup}>
                                <label>Connection Name *</label>
                                <input
                                    type="text"
                                    value={formData.name}
                                    onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                                    placeholder="My Database"
                                    required
                                />
                            </div>
                            <div className={styles.formGroup}>
                                <label>Database Type *</label>
                                <select
                                    value={formData.connector_type}
                                    onChange={(e) => handleTypeChange(e.target.value)}
                                >
                                    {CONNECTION_TYPES.map(type => (
                                        <option key={type.value} value={type.value}>
                                            {type.label}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        {formData.connector_type !== 'sqlite' && formData.connector_type !== 'bigquery' && (
                            <div className={styles.formRow}>
                                <div className={styles.formGroup}>
                                    <label>Host</label>
                                    <input
                                        type="text"
                                        value={formData.host}
                                        onChange={(e) => setFormData(prev => ({ ...prev, host: e.target.value }))}
                                        placeholder="localhost or db.example.com"
                                    />
                                </div>
                                <div className={styles.formGroup}>
                                    <label>Port</label>
                                    <input
                                        type="number"
                                        value={formData.port}
                                        onChange={(e) => setFormData(prev => ({ ...prev, port: parseInt(e.target.value) }))}
                                    />
                                </div>
                            </div>
                        )}

                        <div className={styles.formRow}>
                            <div className={styles.formGroup}>
                                <label>Database Name *</label>
                                <input
                                    type="text"
                                    value={formData.database}
                                    onChange={(e) => setFormData(prev => ({ ...prev, database: e.target.value }))}
                                    placeholder="my_database"
                                    required
                                />
                            </div>
                        </div>

                        {formData.connector_type !== 'sqlite' && (
                            <div className={styles.formRow}>
                                <div className={styles.formGroup}>
                                    <label>Username</label>
                                    <input
                                        type="text"
                                        value={formData.username}
                                        onChange={(e) => setFormData(prev => ({ ...prev, username: e.target.value }))}
                                        placeholder="db_user"
                                    />
                                </div>
                                <div className={styles.formGroup}>
                                    <label>Password</label>
                                    <input
                                        type="password"
                                        value={formData.password}
                                        onChange={(e) => setFormData(prev => ({ ...prev, password: e.target.value }))}
                                        placeholder="••••••••"
                                    />
                                </div>
                            </div>
                        )}

                        <div className={styles.formRow}>
                            <label className={styles.checkbox}>
                                <input
                                    type="checkbox"
                                    checked={formData.ssl}
                                    onChange={(e) => setFormData(prev => ({ ...prev, ssl: e.target.checked }))}
                                />
                                Use SSL/TLS
                            </label>
                        </div>

                        <div className={styles.formActions}>
                            <button type="button" onClick={() => { setShowForm(false); resetForm(); }} className={styles.cancelBtn}>
                                Cancel
                            </button>
                            <button type="submit" disabled={saving} className={styles.saveBtn}>
                                {saving ? <IconLoader size={18} className={styles.spinning} /> : <IconCheck size={18} />}
                                {saving ? 'Saving...' : 'Create Connection'}
                            </button>
                        </div>
                    </form>
                </motion.div>
            )}

            {/* Connections List */}
            {loading ? (
                <div className={styles.loading}>
                    <IconLoader size={32} className={styles.spinning} />
                    <p>Loading connections...</p>
                </div>
            ) : (
                <div className={styles.connectionsList}>
                    {connections.map((conn, index) => (
                        <motion.div
                            key={conn.id}
                            className={styles.connectionCard}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.05 }}
                        >
                            <div className={styles.connIcon}>
                                <IconServer size={24} />
                            </div>
                            <div className={styles.connInfo}>
                                <h3>{conn.name}</h3>
                                <p>{conn.connector_type} • {conn.host || conn.database}</p>
                            </div>
                            <div className={styles.connStatus}>
                                {testResult[conn.id] === 'success' && (
                                    <span className={styles.success}><IconCheck size={16} /> Connected</span>
                                )}
                                {testResult[conn.id] === 'failed' && (
                                    <span className={styles.failed}><IconX size={16} /> Failed</span>
                                )}
                            </div>
                            <div className={styles.connActions}>
                                <button
                                    onClick={() => testConnection(conn.id)}
                                    disabled={testing === conn.id}
                                    className={styles.testBtn}
                                >
                                    {testing === conn.id ? <IconLoader size={16} className={styles.spinning} /> : 'Test'}
                                </button>
                                <button
                                    onClick={() => deleteConnection(conn.id)}
                                    className={styles.deleteBtn}
                                >
                                    <IconTrash size={16} />
                                </button>
                            </div>
                        </motion.div>
                    ))}
                </div>
            )}

            {!loading && connections.length === 0 && !showForm && (
                <div className={styles.empty}>
                    <IconDatabase size={48} />
                    <h3>No connections yet</h3>
                    <p>Add a database connection to query external data</p>
                    <button onClick={() => setShowForm(true)} className={styles.emptyBtn}>
                        <IconPlus size={18} />
                        Add Connection
                    </button>
                </div>
            )}
        </div>
    );
}
