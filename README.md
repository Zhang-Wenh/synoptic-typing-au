# Australian synoptic weather typing

Classifying daily atmospheric circulation over the Australian region from
ERA5, and using the classification to separate two causes of change in
southeast Australian rainfall and extreme heat: how often each weather type
occurs, and how intense conditions are within each type.

**Status: data layer under construction.** Stage 0 of 5. See
[Roadmap](#roadmap).

---

## The question

Long-term change in a regional climate variable can arise two ways. The
circulation patterns that deliver rain or heat can become more or less
frequent, or the conditions associated with a given pattern can intensify
while its frequency stays the same. A trend in the seasonal mean says nothing
about which is happening, and the two have different implications for how much
confidence to place in model projections.

Writing the change in a regional mean as a sum over weather types *k*:

```
Δȳ  =  Σ Δf_k · ȳ_k     (frequency term: circulation occurs differently)
     + Σ f_k · Δȳ_k     (intensity term: same circulation, different outcome)
     + interaction
```

The first term is dynamic and is where models disagree most. The second is
closer to thermodynamic and is better constrained. Splitting an observed trend
between them is more informative than the trend alone.

Three stages follow from this:

1. **Classify.** Define weather types from ERA5 mean sea level pressure and
   500 hPa geopotential height anomalies over 1979–2025.
2. **Decompose.** Attribute observed change in southeast Australian rainfall
   and hot days to the frequency and intensity terms.
3. **Evaluate.** Project CMIP6 circulation onto the observed type centroids
   and test whether models reproduce the observed type frequencies.

---

## Data

All three sources are publicly readable without an account or an institutional
allocation.

| Source | Use | Access |
|---|---|---|
| [ARCO-ERA5](https://github.com/google-research/arco-era5) | Circulation fields | Anonymous read from Google Cloud, Zarr |
| [SILO](https://www.longpaddock.qld.gov.au/silo/) | Observed rainfall and temperature | Public S3, `ap-southeast-2`, CC BY 4.0 |
| [Pangeo CMIP6](https://pangeo-data.github.io/pangeo-cmip6-cloud/) | Model circulation | Anonymous read from Google Cloud, Zarr |

Roughly 150 GB in total after subsetting. Nothing is committed to this
repository; `scripts/` regenerates all of it.

---

## Design decisions

These are the choices that shape the result. Each is a judgement, not a
default, and each is open to challenge.

### Analysis begins in 1979, not 1940

ERA5 extends back to 1940, and ARCO exposes the whole record. The back
extension is **not** used in the main analysis.

Before 1979 there were very few observations over the Southern Ocean and the
Australian region. A reanalysis in an unobserved region is close to a free
model run, and the Southern Hemisphere pre-satellite record carries known
spurious trends. For a study whose central quantity is a *change in weather
type frequency*, that is disqualifying: an apparent frequency trend could be
an artefact of the observing system rather than a feature of the atmosphere.

The 1940–1978 period is examined separately as a sensitivity check, and the
comparison is reported rather than hidden.

### Rainfall from SILO rather than AGCD

Both interpolate the same Bureau of Meteorology station network; the
interpolation methods differ. SILO is chosen because it is openly downloadable
under CC BY 4.0 with no account, and hosted in Sydney. AGCD requires NCI
access, which not every user of this repository will have.

The cost is that results are conditional on SILO's interpolation. Areas with
sparse station coverage — much of inland Australia — carry more uncertainty
than the gridded product's continuous appearance suggests. The target region
is coastal southeast Australia, where station density is highest.

### Types are defined on detrended anomalies

If a long-term trend is left in the field, clustering can separate early from
late periods rather than one circulation pattern from another, and the
subsequent analysis of frequency change becomes circular. Types are therefore
defined on detrended, deseasonalised anomalies, and trends are examined
afterwards in the type frequency time series.

### The typing domain is larger than the target region

Weather systems that determine rainfall in Victoria have centres of action
well outside it — cutoff lows to the southwest, blocking highs in the Tasman.
Classification uses 90–180°E, 10–60°S; impacts are evaluated over southeast
Australia only.

### Seasonal cycle removed by harmonic fit

A day-of-year climatology estimates each calendar day from as many samples as
there are years, which is noisy. Fitting the first three annual harmonics
gives a smooth cycle without absorbing synoptic-scale variance.

---

## Reproducing

```bash
git clone https://github.com/Zhang-Wenh/synoptic-typing-au.git
cd synoptic-typing-au

conda env create -f environment.yml
conda activate climate

# Point config/paths.yaml at wherever the data should live, then:
python scripts/inspect_arco.py           # confirm store coverage and names
python scripts/fetch_era5.py --dry-run   # transfer estimate
python scripts/fetch_era5.py             # the slow step
python scripts/fetch_silo.py

python -m pytest
```

`config/paths.yaml` is the only file that needs editing on a new machine.
Nothing under `src/` hardcodes a filesystem location.

Both fetch scripts are resumable. `fetch_era5.py` writes one Zarr per
variable-year and skips what exists; `fetch_silo.py` resumes partial HTTP
downloads and verifies against `Content-Length` before marking a file
complete.

---

## Layout

```
config/       paths, domain, data sources. The only machine-specific files.
src/
  io/         arco.py, silo.py, cmip6.py  - data access
  preprocess/ weights.py, anomaly.py      - weighting, deseasonalising, detrending
  cluster/    EOF reduction, k-means, stability      [stage 2]
  attribute/  frequency / intensity decomposition    [stage 3]
  evaluate/   CMIP6 projection and bias metrics      [stage 4]
  viz/        cartopy composites
scripts/      command-line entry points
tests/        run against synthetic data, no network
```

---

## Notes on the data

Points that cost time to discover, recorded so they cost no one else any.

**ARCO time axis.** The store's time dimension is longer than its valid
coverage. Always select by date; never by integer index.

**ERA5 latitude descends.** Latitude runs 90 to −90, so a slice must be given
north-first. `slice(-60, -10)` silently returns an empty array rather than
raising. `src/io/arco.py` detects the ordering and handles both.

**ERA5 longitude runs 0–360**, not −180–180. A negative western bound selects
nothing.

**Accumulated variables are stamped at the end of their accumulation period.**
Precipitation and fluxes represent the preceding hour. Aligning them naively
with instantaneous fields introduces a systematic one-hour offset. In ARCO
these variables live in the `co/single-level-forecast` store, not the
reanalysis one.

**CMIP6 calendars differ** between models — 360-day, no-leap, and proleptic
Gregorian all appear. `use_cftime=True` is mandatory, and day-of-year
arithmetic must tolerate a 360-day year.

**Encoding does not survive zarr v2 to v3.** ARCO arrays carry a
`numcodecs.Blosc` compressor from the zarr v2 era. Passing that encoding
through to a new store raises `Expected a BytesBytesCodec` under zarr-python
v3. `src/io/arco.py` clears encoding before writing and lets zarr apply its
own defaults.

**Chunk along time, not space.** Every downstream reduction runs over time
across the whole field. Chunking space would force a rechunk before the first
EOF.

**sqrt(cos φ), not cos φ, before EOF.** EOF decomposes variance; weighting the
data by the square root weights the variance by the cosine. Using cos φ
directly weights variance by cos² and over-represents low latitudes. This
produces plausible-looking output with a systematic bias, so it is guarded by
a test.

---

## Roadmap

| Stage | Content | Status |
|---|---|---|
| 0 | Environment, storage, data access | in progress |
| 1 | Preprocessing to analysis-ready Zarr | |
| 2 | Weather type classification | |
| 3 | Frequency / intensity decomposition | |
| 4 | CMIP6 evaluation | |
| 5 | Write-up and figures | |

---

## License

MIT for the code. The data carry their own licences: ERA5 under the Copernicus
licence, SILO under CC BY 4.0, and CMIP6 per-model terms.
