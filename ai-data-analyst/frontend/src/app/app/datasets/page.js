'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { IconUpload, IconPlus, IconSearch, IconFolder, IconDatabase, IconArrowRight, IconX, IconCheck, IconLoader } from '@/components/icons';
import api from '@/lib/api';
import styles from './page.module.css';

export default function DatasetsPage() {
    const [datasets, setDatasets] = useState([]);
    const [loading, setLoading] = useState(true);
    const [loadError, setLoadError] = useState(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [dragActive, setDragActive] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadError, setUploadError] = useState(null);
    const [uploadSuccess, setUploadSuccess] = useState(null);
    const fileInputRef = useRef(null);

    // Load datasets on mount
    useEffect(() => {
        loadDatasets();
    }, []);

    const loadDatasets = async () => {
        try {
            setLoading(true);
            setLoadError(null);
            const response = await api.listDatasets(1, 50, searchQuery);
            setDatasets(response.items || []);
        } catch (error) {
            console.error('Failed to load datasets:', error);
            setLoadError(error?.message || 'Failed to load datasets');
            setDatasets([]);
        } finally {
            setLoading(false);
        }
    };

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const handleDrop = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        const files = e.dataTransfer?.files;
        if (files && files.length > 0) {
            await uploadFile(files[0]);
        }
    };

    const handleFileSelect = async (e) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            await uploadFile(files[0]);
        }
    };

    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    const waitForDatasetProcessed = async (datasetId, { timeoutMs = 2 * 60 * 1000 } = {}) => {
        const deadline = Date.now() + timeoutMs;
        let attempt = 0;

        while (Date.now() < deadline) {
            const dataset = await api.getDataset(datasetId);
            const status = String(dataset?.status || '').toLowerCase();

            if (status === 'ready') return dataset;
            if (status === 'error') {
                const msg = dataset?.error_message || 'Dataset processing failed';
                throw new Error(msg);
            }

            attempt += 1;
            const delay = Math.min(2500, 400 + attempt * 200);
            await sleep(delay);
        }

        throw new Error('Timed out waiting for dataset processing to complete');
    };

    const uploadFile = async (file) => {
        // Validate file type
        const validTypes = ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.tsv'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!validTypes.includes(ext)) {
            setUploadError(`Invalid file type. Supported: ${validTypes.join(', ')}`);
            return;
        }

        // Validate file size (1GB max)
        if (file.size > 1024 * 1024 * 1024) {
            setUploadError('File too large. Maximum size is 1GB.');
            return;
        }

        try {
            setUploading(true);
            setUploadError(null);
            setUploadProgress(0);

            // Simulate progress (XHR progress would be better; fetch doesn't support it).
            const progressInterval = setInterval(() => {
                setUploadProgress(prev => Math.min(prev + 10, 90));
            }, 200);

            const uploaded = await api.uploadDataset(file, file.name);
            const datasetId = uploaded?.dataset_id;

            if (datasetId) {
                setUploadSuccess(`Uploaded ${file.name}. Processing started...`);
            } else {
                setUploadSuccess(`Successfully uploaded ${file.name}`);
            }

            clearInterval(progressInterval);
            setUploadProgress(100);

            // Refresh dataset list
            await loadDatasets();

            if (datasetId) {
                const processed = await waitForDatasetProcessed(datasetId);
                setUploadSuccess(
                    `Processed ${processed?.name || file.name} (${(processed?.row_count || 0).toLocaleString()} rows, ${processed?.column_count || 0} columns)`
                );
                await loadDatasets();
            }

            // Clear success message after 3 seconds
            setTimeout(() => setUploadSuccess(null), 3000);

        } catch (error) {
            setUploadError(error.message || 'Upload failed. Please try again.');
        } finally {
            setUploading(false);
            setUploadProgress(0);
        }
    };

    const formatFileSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
    };

    const formatDate = (dateStr) => {
        const date = new Date(dateStr);
        const now = new Date();
        const diff = now - date;

        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)} min ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)} hours ago`;
        if (diff < 604800000) return `${Math.floor(diff / 86400000)} days ago`;
        return date.toLocaleDateString();
    };

    const filteredDatasets = datasets.filter(d =>
        String(d?.name || '').toLowerCase().includes(searchQuery.toLowerCase())
    );

    const renderDatasetMeta = (dataset) => {
        const status = String(dataset?.status || '').toLowerCase();

        if (status === 'ready') {
            return `${(dataset.row_count || 0).toLocaleString()} rows | ${dataset.column_count || 0} columns | ${formatFileSize(dataset.file_size_bytes || 0)}`;
        }

        if (status === 'processing') {
            return `Processing... | ${formatFileSize(dataset.file_size_bytes || 0)}`;
        }

        if (status === 'pending') {
            return `Queued... | ${formatFileSize(dataset.file_size_bytes || 0)}`;
        }

        if (status === 'error') {
            return `Error | ${dataset.error_message || 'Processing failed'}`;
        }

        return `${formatFileSize(dataset.file_size_bytes || 0)}`;
    };

    return (
        <div className={styles.page}>
            {/* Header */}
            <div className={styles.header}>
                <div>
                    <h1 className={styles.title}>Datasets</h1>
                    <p className={styles.subtitle}>Manage and analyze your data sources</p>
                </div>
                <button
                    className={styles.uploadBtn}
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                >
                    <IconUpload size={18} />
                    Upload Dataset
                </button>
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv,.xlsx,.xls,.json,.parquet,.tsv"
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                />
            </div>

            {/* Success/Error Messages */}
            {uploadSuccess && (
                <motion.div
                    className={styles.successMessage}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <IconCheck size={18} />
                    {uploadSuccess}
                </motion.div>
            )}
            {uploadError && (
                <motion.div
                    className={styles.errorMessage}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <IconX size={18} />
                    {uploadError}
                    <button onClick={() => setUploadError(null)}>Dismiss</button>
                </motion.div>
            )}

            {/* Upload Zone */}
            <div
                className={`${styles.uploadZone} ${dragActive ? styles.active : ''} ${uploading ? styles.uploading : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => !uploading && fileInputRef.current?.click()}
            >
                {uploading ? (
                    <>
                        <div className={styles.uploadIcon}>
                            <IconLoader size={32} className={styles.spinning} />
                        </div>
                        <p className={styles.uploadText}>Uploading... {uploadProgress}%</p>
                        <div className={styles.progressBar}>
                            <div
                                className={styles.progressFill}
                                style={{ width: `${uploadProgress}%` }}
                            />
                        </div>
                    </>
                ) : (
                    <>
                        <div className={styles.uploadIcon}>
                            <IconUpload size={32} />
                        </div>
                        <p className={styles.uploadText}>
                            Drag and drop files here, or <span>browse</span>
                        </p>
                        <p className={styles.uploadHint}>
                            Supports CSV, Excel (.xlsx, .xls), JSON, Parquet up to 1GB
                        </p>
                    </>
                )}
            </div>

            {/* Search */}
            <div className={styles.searchBar}>
                <IconSearch size={18} className={styles.searchIcon} />
                <input
                    type="text"
                    placeholder="Search datasets..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className={styles.searchInput}
                />
            </div>

            {/* Dataset List */}
            {loading ? (
                <div className={styles.loading}>
                    <IconLoader size={32} className={styles.spinning} />
                    <p>Loading datasets...</p>
                </div>
            ) : (
                <div className={styles.datasetList}>
                    {loadError && (
                        <motion.div
                            className={styles.errorMessage}
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                        >
                            <IconX size={18} />
                            {loadError}
                            <button onClick={() => loadDatasets()}>Retry</button>
                        </motion.div>
                    )}
                    {filteredDatasets.map((dataset, index) => (
                        <motion.div
                            key={dataset.id}
                            className={styles.datasetCard}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.05 }}
                        >
                            <div className={styles.datasetIcon}>
                                <IconDatabase size={20} />
                            </div>
                            <div className={styles.datasetInfo}>
                                <h3 className={styles.datasetName}>{dataset.name}</h3>
                                <p className={styles.datasetMeta}>
                                    {renderDatasetMeta(dataset)}
                                </p>
                            </div>
                            <span className={styles.datasetTime}>{formatDate(dataset.updated_at)}</span>
                            {String(dataset?.status || '').toLowerCase() === 'ready' ? (
                                <Link href={`/app/analysis/new?dataset=${dataset.id}`} className={styles.datasetAction}>
                                    <IconArrowRight size={18} />
                                </Link>
                            ) : (
                                <span className={styles.datasetAction} aria-disabled="true" title="Dataset is not ready yet">
                                    <IconLoader size={18} className={styles.spinning} />
                                </span>
                            )}
                        </motion.div>
                    ))}
                </div>
            )}

            {!loading && filteredDatasets.length === 0 && (
                <div className={styles.empty}>
                    <IconFolder size={48} />
                    <h3>No datasets found</h3>
                    <p>Upload your first dataset to get started</p>
                </div>
            )}
        </div>
    );
}
