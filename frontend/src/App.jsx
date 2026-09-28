import { StrictMode, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import Plotly from 'plotly.js-dist-min';
import './index.css';

const colors = ['#167568', '#bc7854', '#7b83b5', '#c3a349', '#527f9d'];
const metrics = [
  ['quantization_error', 'Quantization error', 'Average distance to the best-matching neuron. Lower is better.'],
  ['topographic_error', 'Topographic error', 'Fraction with non-neighboring first and second matches. Lower is better.'],
  ['silhouette_score', 'Silhouette score', 'Separation of sample clusters. Higher is better.'],
  ['adjusted_rand_index', 'Adjusted Rand index', 'Agreement with diagnosis labels after training. Higher is better.'],
];
const format = (value) => Number.isFinite(value) ? value.toFixed(4) : '—';
const featureLabel = (name) => name.replaceAll('_', ' ').replace(/([123])$/, (_, n) => ` ${['mean', 'SE', 'worst'][Number(n) - 1]}`);

async function request(path, train = false) {
  const response = await fetch(`/api/${path}`, {
    ...(train && { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }),
    signal: AbortSignal.timeout(180000),
  });
  if (path === 'som' && response.status === 404) return null;
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : `Could not load ${path} (${response.status}). Check that the backend is running on port 8000, then refresh.`);
  }
  return response.json();
}

function Chart({ label, data, layout = {} }) {
  const ref = useRef(null);
  useEffect(() => {
    const node = ref.current;
    Plotly.react(node, data, {
      autosize: true, height: 310, margin: { t: 15, r: 20, b: 45, l: 55 },
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: { family: 'Arial, sans-serif', size: 11, color: '#66716c' },
      ...layout,
    }, { displayModeBar: false, responsive: true });
    const observer = new ResizeObserver(() => {
      // A queued resize can finish after React removes the chart.
      if (node.data && node.offsetWidth) Plotly.Plots.resize(node).catch(() => {});
    });
    observer.observe(node);
    return () => { observer.disconnect(); Plotly.purge(node); };
  }, [data, layout]);
  return <div className="chart" ref={ref} role="img" aria-label={label} />;
}

function App() {
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState('loading');
  const [error, setError] = useState('');
  const [feature, setFeature] = useState('');
  const pending = useRef(false);

  async function update(train = false) {
    if (pending.current) return;
    pending.current = true;
    setBusy(train ? 'training' : 'loading');
    setError('');
    try {
      if (train) await request('train', true);
      const [dataset, som] = await Promise.all([request('dataset'), request('som')]);
      const [clusters, evaluation] = som ? await Promise.all([request('clusters'), request('evaluation')]) : [];
      setData({ dataset, som, clusters, evaluation });
    } catch (err) {
      setError(err.name === 'TimeoutError' ? 'The request took too long. Refresh to check whether training finished.'
        : err instanceof TypeError ? 'Cannot reach the backend. Make sure it is running on port 8000, then refresh.' : err.message);
    } finally {
      pending.current = false;
      setBusy('');
    }
  }

  useEffect(() => { update(); }, []);
  const { dataset, som, clusters, evaluation } = data || {};
  const features = Object.keys(clusters?.feature_profiles[0]?.mean_features || {});
  const selectedFeature = features.includes(feature) ? feature : features[0];
  const trained = Boolean(som);

  return <>
    <header className="topbar">
      <a className="brand" href="#overview"><span className="brand-mark">S</span> SOM<span className="brand-light"> / lab</span></a>
      {trained && <nav aria-label="Page sections"><a href="#maps">Maps</a><a href="#clusters">Clusters</a><a href="#evaluation">Evaluation</a></nav>}
      <span className="tag">Research workspace</span>
    </header>

    <main id="overview" aria-busy={Boolean(busy)}>
      <section className="hero">
        <div><p className="eyebrow">SELF-ORGANIZING MAPS · WISCONSIN DATASET</p>
          <h1>Medical data, mapped.</h1>
          <p className="intro">Discover patterns in breast cancer measurements.<br />Train the map, explore the groups, understand the results.</p>
        </div>
        <div className="actions">
          <button className="primary" disabled={Boolean(busy)} onClick={() => update(true)}>{busy === 'training' ? 'Training…' : trained ? 'Retrain model' : 'Train model'}<span aria-hidden="true">↗</span></button>
          <button disabled={Boolean(busy)} onClick={() => update()}>Refresh</button>
        </div>
      </section>

      <div className="status-line" role="status" aria-live="polite">
        <span className={`dot ${busy ? 'working' : error ? 'failed' : ''}`} />
        {busy === 'training' ? 'Training the 9 × 9 map · 10,000 iterations · please keep this page open.'
          : busy ? 'Connecting and loading results…' : error ? 'Could not update the workspace'
          : trained ? `${som.grid.replace('x', ' × ')} map ready · ${clusters.cluster_count} clusters · ${som.n_neurons} neurons` : 'Ready to begin · 9 × 9 map · 3 clusters · seed 42'}
        <span className="status-note">Exploratory research · not a diagnostic tool</span>
      </div>
      {error && <div className="error" role="alert"><strong>{error}</strong>{data && <p>Previously loaded data is still shown. Use Refresh to try again.</p>}</div>}

      {dataset && <section className="dataset-strip" aria-label="Dataset overview">
        {[['Samples', dataset.sample_count], ['Features', dataset.feature_count], ['Benign', dataset.diagnosis_distribution.B], ['Malignant', dataset.diagnosis_distribution.M]].map(([label, value]) =>
          <div key={label}><span>{label}</span><strong>{value?.toLocaleString() ?? '—'}</strong></div>)}
        <p>30 measurements. One shared map.<br /><span>Diagnosis labels are excluded from model fitting.</span></p>
      </section>}

      {!trained && !busy && !error && <section className="empty">
        <span className="empty-symbol" aria-hidden="true">⌘</span><h2>Your exploration starts here.</h2>
        <p>Select <strong>Train model</strong> to group similar samples and reveal the results below.</p>
        <small>Standardize features → train SOM → cluster neurons → evaluate</small>
      </section>}

      {trained && <>
        <section id="maps">
          <div className="section-heading"><div><p className="eyebrow">01 / THE MAP</p><h2>A neighborhood of similar samples</h2></div><span className="tag">Hover to inspect a neuron</span></div>
          <div className="two-columns">
            {[
              ['Neuron distances', 'Darker cells mark larger distances between neighboring neurons.', som.umatrix, [[0, '#eff5ee'], [0.5, '#79b2a1'], [1, '#164c43']], 'Normalized distance'],
              ['Sample density', 'Each cell counts the samples assigned to that neuron.', som.hit_map, [[0, '#faf4eb'], [0.5, '#dfb07f'], [1, '#8c5136']], 'Samples'],
            ].map(([title, description, matrix, colorscale, unit]) => <article className="panel" key={title}>
              <h3>{title}</h3><p>{description}</p>
              <Chart label={title} data={[{ z: matrix, type: 'heatmap', colorscale, xgap: 2, ygap: 2,
                colorbar: { thickness: 9, len: 0.8 }, hovertemplate: `Row %{y} · column %{x}<br>${unit}: %{z}<extra></extra>` }]}
                layout={{ xaxis: { title: { text: 'Neuron column' }, dtick: 1, constrain: 'domain' }, yaxis: { title: { text: 'Neuron row' }, dtick: 1, autorange: 'reversed', scaleanchor: 'x' } }} />
            </article>)}
          </div>
        </section>

        <section id="clusters">
          <div className="section-heading"><div><p className="eyebrow">02 / THE GROUPS</p><h2>Compare the discovered clusters</h2></div></div>
          <div className="panel table-wrap"><table>
            <caption>Diagnosis composition, compared after clustering. Percentages describe this dataset.</caption>
            <thead><tr>{['Cluster', 'Samples', 'Cohort share', 'Benign', 'Malignant', 'Malignant share'].map(label => <th key={label} scope="col">{label}</th>)}</tr></thead>
            <tbody>{clusters.cluster_sizes.map((item, index) => {
              const diagnosis = clusters.diagnosis_distribution.find(row => row.cluster_id === item.cluster_id);
              return <tr key={item.cluster_id}><th scope="row"><i className="cluster-dot" style={{ background: colors[index % colors.length] }} />Cluster {item.cluster_id}</th>
                <td>{item.count}</td><td>{item.percentage}%</td><td>{diagnosis?.benign ?? '—'}</td><td>{diagnosis?.malignant ?? '—'}</td>
                <td><span className="share"><meter min="0" max="100" value={diagnosis?.pct_malignant ?? 0} aria-label={`Cluster ${item.cluster_id} malignant percentage`} />{diagnosis?.pct_malignant ?? '—'}%</span></td></tr>;
            })}</tbody>
          </table></div>
          <article className="panel feature-panel">
            <div className="panel-heading"><div><h3>What makes each group different?</h3><p>Compare one measurement at a time, in its original units.</p></div>
              <label>Measurement<select value={selectedFeature || ''} onChange={event => setFeature(event.target.value)}>{features.map(name => <option key={name} value={name}>{featureLabel(name)}</option>)}</select></label>
            </div>
            <Chart label={`Mean ${featureLabel(selectedFeature || '')} by cluster`} data={[{
              x: clusters.feature_profiles.map(row => `Cluster ${row.cluster_id}`), y: clusters.feature_profiles.map(row => row.mean_features[selectedFeature]),
              type: 'bar', marker: { color: clusters.feature_profiles.map((_, i) => colors[i % colors.length]) },
              hovertemplate: '%{x}<br>Mean: %{y:.4f}<extra></extra>',
            }]} layout={{ height: 250, bargap: 0.65, yaxis: { title: { text: `Mean ${featureLabel(selectedFeature || '')}` }, gridcolor: '#edf0ec', rangemode: 'tozero' } }} />
          </article>
        </section>

        <section id="evaluation">
          <div className="section-heading"><div><p className="eyebrow">03 / THE RESULTS</p><h2>How well does the model fit?</h2></div><span className="tag">Current run</span></div>
          <div className="metric-grid">{metrics.map(([key, label, hint]) => <article className="metric" key={key}><h3>{label}</h3><strong>{format(evaluation[key])}</strong><p>{hint}</p></article>)}</div>
          <article className="panel"><h3>Learning over time</h3><p>Quantization error at each recorded training step. A lower value means a closer fit.</p>
            <Chart label="Quantization error over training iterations" data={[{
              x: evaluation.qe_history.map(row => row.step), y: evaluation.qe_history.map(row => row.qe), type: 'scatter', mode: 'lines',
              line: { color: colors[0], width: 2.5 }, fill: 'tozeroy', fillcolor: '#1675680d', hovertemplate: 'Step %{x}<br>Error: %{y:.4f}<extra></extra>',
            }]} layout={{ height: 250, xaxis: { title: { text: 'Training step' }, gridcolor: '#edf0ec' }, yaxis: { title: { text: 'Quantization error' }, gridcolor: '#edf0ec' } }} />
          </article>
          {evaluation.grid_comparison?.length > 0 && <details className="panel benchmark"><summary>Compare saved grid experiments <span>9 × 9 / 11 × 11 / 13 × 13</span></summary>
            <p>Historical benchmarks, separate from the current run. Selection used a composite of QE, TE, silhouette, and diagnosis-based ARI.</p>
            <div className="table-wrap"><table><thead><tr>{['Grid', 'K', 'QE ↓', 'TE ↓', 'Silhouette ↑', 'ARI ↑', 'Composite ↑'].map(label => <th key={label} scope="col">{label}</th>)}</tr></thead>
              <tbody>{evaluation.grid_comparison.map(row => <tr key={row.grid}><th scope="row">{row.grid}</th><td>{row.k}</td>{['quantization_error', 'topographic_error', 'silhouette_score', 'ari', 'composite_score'].map(key => <td key={key}>{format(row[key])}</td>)}</tr>)}</tbody>
            </table></div>
          </details>}
        </section>
      </>}
      <footer><span>SOM / lab <span className="brand-light">· Medical data clustering</span></span><span>StandardScaler → SOM → K-Means</span></footer>
    </main>
  </>;
}

const root = createRoot(document.getElementById('root'));
root.render(<StrictMode><App /></StrictMode>);
if (import.meta.hot) import.meta.hot.dispose(() => root.unmount());
