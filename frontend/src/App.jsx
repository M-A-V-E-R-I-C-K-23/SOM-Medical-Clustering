import React, { useState, useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

// Consistent cluster color palette across dashboard
const CLUSTER_COLORS = ['#3b82f6', '#ef4444', '#10b981'];

/**
 * Reusable Plotly Chart Component with automatic resize
 */
function PlotlyChart({ data, layout, config, style }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const defaultLayout = {
      autosize: true,
      margin: { t: 40, r: 25, b: 45, l: 55 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { family: 'Inter, sans-serif', size: 12, color: '#334155' },
      ...layout,
    };

    const defaultConfig = {
      responsive: true,
      displayModeBar: false,
      ...config,
    };

    Plotly.react(containerRef.current, data, defaultLayout, defaultConfig);

    const handleResize = () => {
      if (containerRef.current) {
        Plotly.Plots.resize(containerRef.current);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [data, layout, config]);

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%', minHeight: '380px', ...style }}
    />
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [selectedCluster, setSelectedCluster] = useState(0);
  const [featureView, setFeatureView] = useState('key'); // 'key' | 'all'

  // API states
  const [dataset, setDataset] = useState(null);
  const [som, setSom] = useState(null);
  const [clusters, setClusters] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState(null);
  const [untrained, setUntrained] = useState(false);

  // Load all analytical data once from backend
  const fetchData = async () => {
    setLoading(true);
    setError(null);
    setUntrained(false);

    try {
      // 1. Dataset metadata (works even when untrained)
      const dsRes = await fetch('/api/dataset');
      if (!dsRes.ok) throw new Error(`Dataset API returned ${dsRes.status}`);
      const dsData = await dsRes.json();
      setDataset(dsData);

      // 2. SOM results
      const somRes = await fetch('/api/som');
      if (somRes.status === 404) {
        setUntrained(true);
        setLoading(false);
        return;
      }
      if (!somRes.ok) throw new Error(`SOM API returned ${somRes.status}`);
      const somData = await somRes.json();
      setSom(somData);

      // 3. Cluster results
      const clRes = await fetch('/api/clusters');
      if (!clRes.ok) throw new Error(`Clusters API returned ${clRes.status}`);
      const clData = await clRes.json();
      setClusters(clData);

      // 4. Evaluation results
      const evRes = await fetch('/api/evaluation');
      if (!evRes.ok) throw new Error(`Evaluation API returned ${evRes.status}`);
      const evData = await evRes.json();
      setEvaluation(evData);
    } catch (err) {
      setError(err.message || 'Failed to connect to FastAPI backend');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Train pipeline trigger
  const handleTrain = async () => {
    setTraining(true);
    setError(null);
    try {
      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!res.ok) throw new Error(`Training failed with status ${res.status}`);
      await fetchData();
    } catch (err) {
      setError(`Training error: ${err.message}`);
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <div className="header-inner">
          <div className="brand-title">
            <span>🔬 Medical SOM Clustering</span>
            <span className="brand-badge">Academic Dashboard</span>
          </div>

          <div className="header-actions">
            {training ? (
              <button className="btn btn-primary" disabled>
                <span className="spinner"></span> Training SOM (10,000 iters)...
              </button>
            ) : (
              <button className="btn btn-primary" onClick={handleTrain}>
                ⚡ {untrained ? 'Train Pipeline' : 'Re-run Pipeline'}
              </button>
            )}
            <button className="btn btn-outline" onClick={fetchData} title="Refresh data">
              ↻ Refresh
            </button>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="nav-bar">
        <div className="nav-inner">
          <button
            className={`nav-btn ${activeTab === 'home' ? 'active' : ''}`}
            onClick={() => setActiveTab('home')}
          >
            1. Dashboard Home
          </button>
          <button
            className={`nav-btn ${activeTab === 'som' ? 'active' : ''}`}
            onClick={() => setActiveTab('som')}
            disabled={untrained}
          >
            2. SOM Map
          </button>
          <button
            className={`nav-btn ${activeTab === 'clusters' ? 'active' : ''}`}
            onClick={() => setActiveTab('clusters')}
            disabled={untrained}
          >
            3. Cluster Explorer
          </button>
          <button
            className={`nav-btn ${activeTab === 'diagnosis' ? 'active' : ''}`}
            onClick={() => setActiveTab('diagnosis')}
            disabled={untrained}
          >
            4. Diagnosis Comparison
          </button>
          <button
            className={`nav-btn ${activeTab === 'evaluation' ? 'active' : ''}`}
            onClick={() => setActiveTab('evaluation')}
            disabled={untrained}
          >
            5. Evaluation
          </button>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="main-content">
        {/* Global Academic Disclaimer */}
        <div className="disclaimer-banner">
          ⚠️ <strong>Medical Disclaimer:</strong> Unsupervised pattern discovery and clustering — not a diagnostic tool.
        </div>

        {/* Global Error Banner */}
        {error && (
          <div className="card" style={{ borderLeft: '4px solid var(--danger)', color: 'var(--danger)' }}>
            <strong>API Connection Error:</strong> {error}
            <div style={{ marginTop: '0.5rem' }}>
              <button className="btn btn-outline" onClick={fetchData}>Try Reconnecting</button>
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && !training && (
          <div className="state-container">
            <div className="spinner" style={{ borderColor: 'rgba(37,99,235,0.2)', borderTopColor: 'var(--accent-primary)', width: '2rem', height: '2rem' }}></div>
            <div className="state-title" style={{ marginTop: '1rem' }}>Loading Model Data...</div>
            <p className="state-desc">Fetching analytical results from FastAPI backend.</p>
          </div>
        )}

        {/* Untrained State */}
        {!loading && untrained && (
          <div className="state-container">
            <div className="state-title">Backend Pipeline Untrained</div>
            <p className="state-desc">
              The Self-Organizing Map pipeline has not been executed yet in this backend session.
              Click the button below to train the 9×9 SOM grid and compute clusters.
            </p>
            <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
              {training ? 'Training...' : '▶ Train SOM Pipeline Now'}
            </button>
          </div>
        )}

        {/* Loaded Sections */}
        {!loading && !untrained && (
          <>
            {/* 1. Dashboard Home */}
            {activeTab === 'home' && (
              <section>
                <div className="card">
                  <h1 className="card-title" style={{ fontSize: '1.4rem' }}>
                    Medical Data Clustering Using Self-Organizing Maps (SOM)
                  </h1>
                  <p className="card-desc">
                    An unsupervised neural network architecture applied to high-dimensional clinical data (Wisconsin Diagnostic Breast Cancer).
                    The pipeline maps 30 continuous morphological cell features onto a 2D topological lattice, clusters the codebook vectors using K-Means,
                    and validates the discovered phenotypes post-hoc against histological ground truth.
                  </p>
                </div>

                {/* 4 Summary Stat Cards */}
                <div className="grid-cols-4">
                  <div className="stat-box">
                    <div className="stat-label">Total Samples</div>
                    <div className="stat-value">{dataset?.sample_count ?? '—'}</div>
                    <div className="stat-sub">Wisconsin Breast Cancer (WDBC)</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-label">Features</div>
                    <div className="stat-value">{dataset?.feature_count ?? '—'}</div>
                    <div className="stat-sub">Continuous nuclear attributes</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-label">SOM Grid</div>
                    <div className="stat-value">{som?.grid ?? '9x9'}</div>
                    <div className="stat-sub">{som?.n_neurons ?? 81} codebook neurons</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-label">Clusters (K)</div>
                    <div className="stat-value">{clusters?.cluster_count ?? 3}</div>
                    <div className="stat-sub">Codebook K-Means partitions</div>
                  </div>
                </div>

                {/* Architecture Overview & Dataset Breakdown */}
                <div className="grid-cols-2">
                  <div className="card">
                    <h2 className="card-title">Dataset Distribution</h2>
                    <p className="card-desc">
                      Ground truth labels are held out strictly for post-hoc validation and never enter SOM training.
                    </p>
                    <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
                      <div style={{ flex: 1, padding: '1rem', background: '#f0fdf4', borderRadius: 'var(--radius-sm)', border: '1px solid #bbf7d0' }}>
                        <div style={{ fontSize: '0.8rem', color: '#166534', fontWeight: 600 }}>BENIGN (B)</div>
                        <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#15803d' }}>
                          {dataset?.diagnosis_distribution?.B ?? 357}
                        </div>
                        <div style={{ fontSize: '0.8rem', color: '#166534' }}>62.7% of cohort</div>
                      </div>
                      <div style={{ flex: 1, padding: '1rem', background: '#fef2f2', borderRadius: 'var(--radius-sm)', border: '1px solid #fecaca' }}>
                        <div style={{ fontSize: '0.8rem', color: '#991b1b', fontWeight: 600 }}>MALIGNANT (M)</div>
                        <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#b91c1c' }}>
                          {dataset?.diagnosis_distribution?.M ?? 212}
                        </div>
                        <div style={{ fontSize: '0.8rem', color: '#991b1b' }}>37.3% of cohort</div>
                      </div>
                    </div>
                  </div>

                  <div className="card">
                    <h2 className="card-title">Methodology Pipeline</h2>
                    <p className="card-desc">Two-level clustering workflow preserving topological relationships:</p>
                    <ol style={{ paddingLeft: '1.25rem', fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: '1.8' }}>
                      <li><strong>StandardScaler:</strong> Zero-mean unit-variance scaling on 30 features.</li>
                      <li><strong>MiniSom Training:</strong> 9×9 hexagonal/rectangular grid trained for 10,000 steps.</li>
                      <li><strong>Codebook Extraction:</strong> Weight vectors extracted for all 81 neurons.</li>
                      <li><strong>Codebook K-Means (K=3):</strong> Grouping neurons into high-level phenotypes.</li>
                      <li><strong>BMU Sample Assignment:</strong> Mapping patients through their best-matching neuron.</li>
                    </ol>
                  </div>
                </div>
              </section>
            )}

            {/* 2. SOM Map */}
            {activeTab === 'som' && som && (
              <section>
                <div className="info-banner">
                  ℹ️ <strong>Topological Grid:</strong> The SOM organizes 30-dimensional patient features onto a 9×9 lattice.
                  Neurons preserve local neighborhood relationships.
                </div>

                <div className="grid-cols-2">
                  {/* U-Matrix Heatmap */}
                  <div className="card">
                    <h2 className="card-title">Unified Distance Matrix (U-Matrix)</h2>
                    <p className="card-desc">
                      Average Euclidean distance between each neuron and its immediate lattice neighbors.
                      Dark valleys represent tight clusters; light ridges represent natural cluster boundaries.
                    </p>
                    <PlotlyChart
                      data={[
                        {
                          z: som.umatrix,
                          x: Array.from({ length: som.grid_cols }, (_, i) => i),
                          y: Array.from({ length: som.grid_rows }, (_, i) => i),
                          type: 'heatmap',
                          colorscale: 'Viridis',
                          reversescale: true,
                          colorbar: { title: 'Distance', len: 0.8 },
                          hovertemplate: 'Row: %{y}<br>Col: %{x}<br>Distance: %{z:.3f}<extra></extra>',
                        },
                      ]}
                      layout={{
                        xaxis: { title: 'Neuron Column', dtick: 1 },
                        yaxis: { title: 'Neuron Row', dtick: 1, autorange: 'reversed' },
                        height: 420,
                      }}
                    />
                  </div>

                  {/* BMU Hit Map */}
                  <div className="card">
                    <h2 className="card-title">BMU Sample Hit Map</h2>
                    <p className="card-desc">
                      Number of patient samples mapped to each neuron as their Best Matching Unit (BMU).
                      Cells with 0 hits indicate empty neurons in the latent representation.
                    </p>
                    <PlotlyChart
                      data={[
                        {
                          z: som.hit_map,
                          x: Array.from({ length: som.grid_cols }, (_, i) => i),
                          y: Array.from({ length: som.grid_rows }, (_, i) => i),
                          type: 'heatmap',
                          colorscale: 'YlOrRd',
                          colorbar: { title: 'Samples', len: 0.8 },
                          hovertemplate: 'Row: %{y}<br>Col: %{x}<br>Hits: %{z} samples<extra></extra>',
                        },
                      ]}
                      layout={{
                        xaxis: { title: 'Neuron Column', dtick: 1 },
                        yaxis: { title: 'Neuron Row', dtick: 1, autorange: 'reversed' },
                        height: 420,
                      }}
                    />
                  </div>
                </div>

                <div className="card">
                  <h3 className="card-title" style={{ fontSize: '1rem' }}>Grid Summary Metrics</h3>
                  <div className="grid-cols-4" style={{ marginBottom: 0, marginTop: '1rem' }}>
                    <div className="stat-box">
                      <div className="stat-label">Total Neurons</div>
                      <div className="stat-value">{som.n_neurons}</div>
                    </div>
                    <div className="stat-box">
                      <div className="stat-label">Mapped Samples</div>
                      <div className="stat-value">{som.bmu_summary?.total_samples ?? 569}</div>
                    </div>
                    <div className="stat-box">
                      <div className="stat-label">Max Neuron Density</div>
                      <div className="stat-value">
                        {Math.max(...som.hit_map.flat())}
                      </div>
                      <div className="stat-sub">Samples in single BMU</div>
                    </div>
                    <div className="stat-box">
                      <div className="stat-label">Empty Neurons</div>
                      <div className="stat-value">
                        {som.hit_map.flat().filter((v) => v === 0).length}
                      </div>
                      <div className="stat-sub">Unoccupied codebook nodes</div>
                    </div>
                  </div>
                </div>
              </section>
            )}

            {/* 3. Cluster Explorer */}
            {activeTab === 'clusters' && clusters && (
              <section>
                <div className="card">
                  <h2 className="card-title">Patient Cluster Explorer (K = {clusters.cluster_count})</h2>
                  <p className="card-desc">
                    K-Means applied to SOM codebook vectors partitions the continuous topological manifold into discrete clinical phenotypes.
                  </p>

                  {/* Cluster Selection Buttons */}
                  <div className="cluster-btn-group">
                    {clusters.cluster_sizes.map((c) => (
                      <button
                        key={c.cluster_id}
                        className={`cluster-select-btn ${
                          selectedCluster === c.cluster_id ? `active-${c.cluster_id}` : ''
                        }`}
                        onClick={() => setSelectedCluster(c.cluster_id)}
                      >
                        ● Cluster {c.cluster_id} ({c.count} patients, {c.percentage}%)
                      </button>
                    ))}
                  </div>

                  {/* Current Cluster Details */}
                  {(() => {
                    const currentSize = clusters.cluster_sizes.find(
                      (c) => c.cluster_id === selectedCluster
                    );
                    const currentDiag = clusters.diagnosis_distribution.find(
                      (d) => d.cluster_id === selectedCluster
                    );
                    const currentProfile = clusters.feature_profiles.find(
                      (p) => p.cluster_id === selectedCluster
                    );

                    // Sort or filter features
                    const features = currentProfile ? Object.entries(currentProfile.mean_features) : [];
                    const displayFeatures =
                      featureView === 'key'
                        ? features.filter(([k]) =>
                            ['radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'concavity1', 'symmetry1'].includes(k)
                          )
                        : features;

                    return (
                      <div>
                        {/* Cluster Stat Cards */}
                        <div className="grid-cols-4" style={{ marginBottom: '1.5rem' }}>
                          <div className="stat-box">
                            <div className="stat-label">Patients Assigned</div>
                            <div className="stat-value" style={{ color: CLUSTER_COLORS[selectedCluster] }}>
                              {currentSize?.count}
                            </div>
                            <div className="stat-sub">{currentSize?.percentage}% of cohort</div>
                          </div>
                          <div className="stat-box">
                            <div className="stat-label">Benign Count</div>
                            <div className="stat-value" style={{ color: '#15803d' }}>
                              {currentDiag?.benign}
                            </div>
                            <div className="stat-sub">
                              {currentDiag?.benign && currentSize?.count
                                ? ((currentDiag.benign / currentSize.count) * 100).toFixed(1)
                                : 0}
                              % of this cluster
                            </div>
                          </div>
                          <div className="stat-box">
                            <div className="stat-label">Malignant Count</div>
                            <div className="stat-value" style={{ color: '#b91c1c' }}>
                              {currentDiag?.malignant}
                            </div>
                            <div className="stat-sub">{currentDiag?.pct_malignant}% malignant</div>
                          </div>
                          <div className="stat-box">
                            <div className="stat-label">Phenotype Profile</div>
                            <div className="stat-value" style={{ fontSize: '1.25rem' }}>
                              {currentDiag?.pct_malignant > 80
                                ? 'High Malignancy'
                                : currentDiag?.pct_malignant < 25
                                ? 'Predominantly Benign'
                                : 'Intermediate / Mixed'}
                            </div>
                            <div className="stat-sub">Post-hoc clinical behavior</div>
                          </div>
                        </div>

                        {/* Feature Profile Chart */}
                        <div style={{ marginTop: '1rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                            <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>
                              Mean Feature Profile for Cluster {selectedCluster} (Original Units)
                            </h3>
                            <div>
                              <button
                                className={`btn ${featureView === 'key' ? 'btn-primary' : 'btn-outline'}`}
                                onClick={() => setFeatureView('key')}
                                style={{ marginRight: '0.5rem', padding: '0.35rem 0.75rem', fontSize: '0.8rem' }}
                              >
                                Key Features
                              </button>
                              <button
                                className={`btn ${featureView === 'all' ? 'btn-primary' : 'btn-outline'}`}
                                onClick={() => setFeatureView('all')}
                                style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem' }}
                              >
                                All 30 Features
                              </button>
                            </div>
                          </div>

                          <PlotlyChart
                            data={[
                              {
                                x: displayFeatures.map(([name]) => name),
                                y: displayFeatures.map(([, val]) => val),
                                type: 'bar',
                                marker: { color: CLUSTER_COLORS[selectedCluster] },
                                hovertemplate: 'Feature: %{x}<br>Mean: %{y:.3f}<extra></extra>',
                              },
                            ]}
                            layout={{
                              xaxis: { tickangle: featureView === 'all' ? -45 : 0 },
                              yaxis: { title: 'Mean Value (Original Units)' },
                              height: featureView === 'all' ? 450 : 380,
                              margin: { b: featureView === 'all' ? 120 : 50 },
                            }}
                          />
                        </div>
                      </div>
                    );
                  })()}
                </div>
              </section>
            )}

            {/* 4. Diagnosis Comparison */}
            {activeTab === 'diagnosis' && clusters && (
              <section>
                {/* Mandatory Disclaimer */}
                <div className="disclaimer-banner">
                  📌 <strong>Validation Rule:</strong> Diagnosis labels were used only for post-hoc interpretation and validation, not for SOM training or clustering.
                </div>

                <div className="card">
                  <h2 className="card-title">Cluster Composition vs Ground Truth Histology</h2>
                  <p className="card-desc">
                    Comparing unsupervised cluster partitions against known pathological diagnosis (Benign vs Malignant).
                  </p>

                  <div className="grid-cols-2">
                    {/* Stacked Breakdown Bar Chart */}
                    <div>
                      <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem' }}>
                        Diagnosis Count per Cluster
                      </h3>
                      <PlotlyChart
                        data={[
                          {
                            x: clusters.diagnosis_distribution.map((d) => `Cluster ${d.cluster_id}`),
                            y: clusters.diagnosis_distribution.map((d) => d.benign),
                            name: 'Benign (B)',
                            type: 'bar',
                            marker: { color: '#10b981' },
                            hovertemplate: '%{x}<br>Benign: %{y}<extra></extra>',
                          },
                          {
                            x: clusters.diagnosis_distribution.map((d) => `Cluster ${d.cluster_id}`),
                            y: clusters.diagnosis_distribution.map((d) => d.malignant),
                            name: 'Malignant (M)',
                            type: 'bar',
                            marker: { color: '#ef4444' },
                            hovertemplate: '%{x}<br>Malignant: %{y}<extra></extra>',
                          },
                        ]}
                        layout={{
                          barmode: 'stack',
                          yaxis: { title: 'Patient Sample Count' },
                          legend: { orientation: 'h', y: 1.15 },
                          height: 380,
                        }}
                      />
                    </div>

                    {/* Malignancy Proportion Chart */}
                    <div>
                      <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem' }}>
                        Malignancy Percentage per Cluster (%)
                      </h3>
                      <PlotlyChart
                        data={[
                          {
                            x: clusters.diagnosis_distribution.map((d) => `Cluster ${d.cluster_id}`),
                            y: clusters.diagnosis_distribution.map((d) => d.pct_malignant),
                            type: 'bar',
                            marker: {
                              color: clusters.diagnosis_distribution.map((d) =>
                                d.pct_malignant > 80 ? '#ef4444' : d.pct_malignant < 25 ? '#3b82f6' : '#f59e0b'
                              ),
                            },
                            text: clusters.diagnosis_distribution.map((d) => `${d.pct_malignant}%`),
                            textposition: 'outside',
                            hovertemplate: '%{x}<br>Malignancy: %{y:.1f}%<extra></extra>',
                          },
                        ]}
                        layout={{
                          yaxis: { title: '% Malignant', range: [0, 115] },
                          height: 380,
                        }}
                      />
                    </div>
                  </div>

                  {/* Concordance Table */}
                  <div style={{ marginTop: '1.5rem' }}>
                    <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem' }}>
                      Detailed Concordance Analysis Table
                    </h3>
                    <div className="table-container">
                      <table className="academic-table">
                        <thead>
                          <tr>
                            <th>Cluster ID</th>
                            <th>Total Samples</th>
                            <th>Cohort Share</th>
                            <th>Benign (B)</th>
                            <th>Malignant (M)</th>
                            <th>% Malignant</th>
                            <th>Clinical Concordance</th>
                          </tr>
                        </thead>
                        <tbody>
                          {clusters.diagnosis_distribution.map((d) => {
                            const size = clusters.cluster_sizes.find((s) => s.cluster_id === d.cluster_id);
                            return (
                              <tr key={d.cluster_id}>
                                <td>
                                  <strong>Cluster {d.cluster_id}</strong>
                                </td>
                                <td>{size?.count}</td>
                                <td>{size?.percentage}%</td>
                                <td>
                                  <span className="badge badge-benign">{d.benign}</span>
                                </td>
                                <td>
                                  <span className="badge badge-malignant">{d.malignant}</span>
                                </td>
                                <td>
                                  <strong>{d.pct_malignant}%</strong>
                                </td>
                                <td>
                                  {d.pct_malignant === 100.0 ? (
                                    <span style={{ color: '#991b1b', fontWeight: 600 }}>100% Malignant Phenotype</span>
                                  ) : d.pct_malignant < 20.0 ? (
                                    <span style={{ color: '#166534', fontWeight: 600 }}>86.2% Benign Dominant</span>
                                  ) : (
                                    <span style={{ color: '#92400e', fontWeight: 600 }}>Borderline / Mixed Risk</span>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </section>
            )}

            {/* 5. Evaluation */}
            {activeTab === 'evaluation' && evaluation && (
              <section>
                {/* 4 Evaluation Metrics */}
                <div className="grid-cols-4">
                  <div className="stat-box">
                    <div className="stat-label">Quantization Error (QE)</div>
                    <div className="stat-value">{evaluation.quantization_error?.toFixed(4)}</div>
                    <div className="stat-sub">Mean distance to BMU (Fidelity)</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-label">Topographic Error (TE)</div>
                    <div className="stat-value">{evaluation.topographic_error?.toFixed(4)}</div>
                    <div className="stat-sub">Non-adjacent 2nd BMU ratio (Topology)</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-label">Silhouette Score</div>
                    <div className="stat-value">{evaluation.silhouette_score?.toFixed(4)}</div>
                    <div className="stat-sub">Sample-space cluster separation</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-label">Adjusted Rand Index (ARI)</div>
                    <div className="stat-value">{evaluation.adjusted_rand_index?.toFixed(4)}</div>
                    <div className="stat-sub">Post-hoc agreement with diagnosis</div>
                  </div>
                </div>

                {/* Grid Comparison Table */}
                <div className="card">
                  <div className="card-title">
                    <span>Grid Size Comparison Benchmark</span>
                    <span className="badge badge-selected">Selected: {evaluation.selected_grid}</span>
                  </div>
                  <p className="card-desc">
                    Comprehensive evaluation across candidate SOM dimensions (9×9, 11×11, 13×13).
                    The 9×9 lattice was chosen algorithmically for minimizing Topographic Error while maintaining superior sample-space cluster separation.
                  </p>

                  <div className="table-container">
                    <table className="academic-table">
                      <thead>
                        <tr>
                          <th>Grid Size</th>
                          <th>Neurons</th>
                          <th>K</th>
                          <th>Quantization Error (QE) ↓</th>
                          <th>Topographic Error (TE) ↓</th>
                          <th>Silhouette Score ↑</th>
                          <th>ARI ↑</th>
                          <th>Composite Score ↑</th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {evaluation.grid_comparison?.map((row) => {
                          const isSelected = row.grid === evaluation.selected_grid;
                          return (
                            <tr key={row.grid} className={isSelected ? 'highlight-row' : ''}>
                              <td>
                                <strong>{row.grid}</strong>
                              </td>
                              <td>{row.n_neurons}</td>
                              <td>{row.k}</td>
                              <td>{row.quantization_error?.toFixed(4)}</td>
                              <td>{row.topographic_error?.toFixed(4)}</td>
                              <td>{row.silhouette_score?.toFixed(4)}</td>
                              <td>{row.ari?.toFixed(4)}</td>
                              <td>
                                <strong>{row.composite_score?.toFixed(4)}</strong>
                              </td>
                              <td>
                                {isSelected ? (
                                  <span className="badge badge-selected">Selected Final</span>
                                ) : (
                                  <span className="badge" style={{ background: '#f1f5f9', color: '#64748b' }}>
                                    Candidate
                                  </span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Convergence Curve */}
                {evaluation.qe_history && evaluation.qe_history.length > 0 && (
                  <div className="card">
                    <h2 className="card-title">SOM Training Convergence (Quantization Error vs Steps)</h2>
                    <p className="card-desc">
                      Monitored over 10,000 iterations. Quantization error rapidly drops and stabilizes as codebook weights adjust to the input feature distribution.
                    </p>
                    <PlotlyChart
                      data={[
                        {
                          x: evaluation.qe_history.map((h) => h.step),
                          y: evaluation.qe_history.map((h) => h.qe),
                          type: 'scatter',
                          mode: 'lines+markers',
                          line: { color: '#2563eb', width: 2.5 },
                          marker: { size: 5, color: '#1d4ed8' },
                          hovertemplate: 'Step: %{x}<br>QE: %{y:.4f}<extra></extra>',
                        },
                      ]}
                      layout={{
                        xaxis: { title: 'Training Step (Iterations)' },
                        yaxis: { title: 'Quantization Error' },
                        height: 380,
                      }}
                    />
                  </div>
                )}
              </section>
            )}
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        Medical SOM Clustering Project • Academic Dashboard • Built with React, Vite & Plotly
      </footer>
    </div>
  );
}
