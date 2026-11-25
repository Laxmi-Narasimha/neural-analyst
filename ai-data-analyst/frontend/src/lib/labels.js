// People-facing names for artifacts. Artifacts are stored as "<operator>:<part>", for example
// "ml_score:predictions" or "resample_aggregate:trend"; show "Predictions · ml score" instead.
function words(text) {
    return String(text || '').replace(/_/g, ' ').trim();
}

export function artifactLabel(name) {
    if (!name) return '';
    const [operator, ...rest] = String(name).split(':');
    const part = words(rest.join(':'));
    if (!part) return words(operator);
    return `${part.charAt(0).toUpperCase()}${part.slice(1)} · ${words(operator)}`;
}

// Answers put one fact per line. Markdown joins single line breaks into one paragraph, so turn
// them into hard breaks, except before Markdown list items and blank lines, which Markdown handles.
export function keepLineBreaks(text) {
    return String(text || '').replace(/([^\n])\n(?!\n|\s*(?:[-*+]|\d+\.)\s)/g, '$1  \n');
}
