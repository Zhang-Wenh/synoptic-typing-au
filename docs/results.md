# Results

Southeast Australia, 1979-2025, from ERA5 circulation and SILO surface
observations. Every number below is reproducible from the scripts in this
repository; the commands that produce each section are given alongside it.

Two things are worth reading before the numbers. The classification does not
find distinct circulation regimes, and says so. The rainfall trends are not
statistically significant, and say so. What the analysis does establish is a
consistent decomposition across four impact indices, validated against a case
where the answer is known in advance.

---

## 1. The classification is a stratification, not a set of regimes

Eight types, fitted by k-means on eight principal components of the joint
mean-sea-level-pressure and 500 hPa geopotential height anomaly field.

```
python scripts/eof.py
python scripts/classify.py --sweep --modes 8 12 16
python scripts/classify.py --k 8 --modes 8
```

Reproducibility across random restarts exceeds red-noise surrogates at k = 8
(observed 0.737 against a surrogate 95th percentile of 0.681, 50 surrogates),
but not at k = 6 or k = 10. Since nine values of k were tested, a single
exceedance is suggestive rather than conclusive.

Four diagnostics indicate the types partition a continuum:

| diagnostic | value | what distinct regimes would give |
|---|---|---|
| frequency range | 10.7 to 14.0 per cent | uneven: a dominant type and rarer ones |
| persistence | 2.25 days, spread 0.20 across types | varies by type; blocking outlasts zonal flow |
| seasonality | flat, 7 to 10 per cent per month | some types confined to one season |
| stability under perturbation | 3.4 per cent perturbation reassigns 33 per cent of days | boundaries are sparsely populated |

The last is the sharpest. Removing the linear trend from the input changes the
anomaly field by 0.26 hPa against a field standard deviation of 7.85 hPa, and
a third of days change type. Separated clusters have few samples near their
boundaries and do not behave this way.

Reading the centroid coordinates directly settles the geometry. All eight lie
almost entirely within the first three principal components. Five are evenly
spaced around a circle in the PC1-PC2 plane at intervals of 68 to 75 degrees,
which are phases of an eastward-propagating wave; two are the positive and
negative phases of PC3; one sits on PC4 with the smallest amplitude.

The leading two EOFs are a quadrature pair. Their lag correlation is exactly
zero at zero lag, antisymmetric on either side, and peaks at plus and minus
two days, giving a period of about eight days and an eastward phase speed near
7 m/s. That is the signature of a synoptic-scale Rossby wave, and it is why
their variances are so close (19.8 and 18.1 per cent): the physical quantity is
their sum.

This is the expected result for this region. The Southern Hemisphere
mid-latitudes here are almost entirely ocean, with no orographic forcing to
anchor the flow in preferred quasi-stationary states.

**Consequence for what follows.** The decomposition needs a stratification
that is stable and reproducible. It does not need the types to be separated by
gaps. The types are described here as a stratification throughout, never as
regimes.

---

## 2. Types are distinguished by advection direction

```
python scripts/composite.py --tag _nd
```

The anomalous meridional geostrophic wind over the target region, computed
from each type's composite pressure field, correlates with that type's mean
daily maximum temperature at r = -0.85 across the eight types. Northerly flow
(negative v, from the continental interior) gives the warmest types; southerly
flow (positive v, from the Southern Ocean) the coolest.

| type | v (m/s) | mean Tmax (degC) | mean rain (mm/day) |
|---|---|---|---|
| 4 | -2.50 | 18.75 | 1.29 |
| 0 | -2.89 | 17.78 | 2.27 |
| 5 | +0.85 | 16.45 | 1.75 |
| 7 | +4.00 | 15.28 | 1.93 |

Rainfall shows no such relationship (r = -0.18). Temperature is controlled by
advection direction; rainfall is controlled by moisture supply and lifting,
which do not follow from the wind direction alone. The two are governed by
different physics, which is why the same circulation change produces frequency
terms of opposite sign for them.

---

## 3. Decomposition of the observed trends

```
python scripts/attribute.py --index rain --seeds 5
python scripts/attribute.py --index tmax --seeds 5
python scripts/attribute.py --index hot  --seeds 5
```

Change over 1979-2025, with the frequency and intensity terms expressed as a
percentage of the observed change:

| index | observed change | frequency | intensity | significant |
|---|---|---|---|---|
| mean daily maximum temperature | +1.28 degC | -9 per cent | +112 per cent | yes |
| hot-day frequency, cool season | +5.1 percentage points | -11 per cent | +125 per cent | yes |
| rainfall, cool season | -17 per cent | -13 per cent | +110 per cent | no |
| rainfall, warm season | +27 per cent | +18 per cent | +42 per cent | no |

Hot days are those above the 90th percentile of the same half of the year, so
that both seasons have exceedances to analyse; a single annual threshold
leaves the cool season with almost none.

The two rainfall seasons move in opposite directions and nearly cancel in the
annual mean, which falls by 0.6 per cent over the whole record. An annual
analysis alone would report no change.

Signs are stable across five refitted partitions in every case but one. The
frequency term for annual hot-day count flips sign between partitions and is
therefore not reported as a result, despite a narrow bootstrap interval. The
total trend is identical across partitions by construction, which serves as a
check on the implementation rather than as a finding.

### The temperature result validates the method

Warming is known in advance to be predominantly thermodynamic: greenhouse
forcing lifts the whole temperature distribution and every circulation type
warms with it. A decomposition that attributed a large share of it to
circulation redistribution would be wrong.

It attributes 112 per cent to the intensity term. The observed +1.28 degC also
matches independently documented warming for the region over this period.

This matters for the rainfall result, where the answer is not known in
advance. The method returns the right answer where the answer is checkable.

### Circulation change alone would have cooled the region slightly

In the non-detrended classification, the frequency term for cool-season
temperature is -0.00286 degC per year, and its bootstrap interval excludes
zero. Over the record that is -0.13 degC against an observed +1.15 degC.

The mechanism is visible in the per-type table. The type whose frequency falls
fastest is the warmest (northerly flow, 18.75 degC); the type whose frequency
rises fastest is among the coolest (southerly flow, 16.45 degC). Circulation
change is reducing warm-advection days and increasing cool-advection ones.

---

## 4. What drives the frequency change is largely unresolved

```
python scripts/sam.py --tag _nd
```

Both a Southern Annular Mode proxy and a subtropical ridge index trend upward
over the record, as documented elsewhere. Regressing each type's yearly
frequency on them and refitting the trend to the residual:

| index | cool-season frequency change remaining | types with abs(r) > 0.5 |
|---|---|---|
| SAM proxy | 69 per cent | 1 of 8 |
| subtropical ridge | 62 per cent | 4 of 8 |

The ridge index relates more closely to the two types that drive the
temperature frequency term (r = -0.68 and +0.64, against -0.48 and +0.38 for
the SAM proxy). But the two indices correlate with each other at 0.77, so this
does not establish a mechanistic distinction, and roughly two thirds of the
frequency change is unexplained by either.

The SAM proxy here is regional and truncated at 60S; the standard definition
uses 65S and the full latitude circle. Any quantitative comparison with
published values needs the Marshall station-based index.

---

## Limitations

**Single reanalysis.** Everything rests on ERA5. Southern Hemisphere
reanalysis trends before the satellite era are known to be unreliable, which
is why the record starts in 1979, but agreement between reanalyses has not
been tested here. No frequency-trend result should be considered established
until it is reproduced in at least one other reanalysis.

**Rainfall trends are not significant.** The cool-season decline and warm-season
increase are described, not claimed. Their bootstrap intervals span zero.

**Detrending changes the frequency term by a factor of four to nine.**
Classifying on detrended anomalies keeps the types from grouping days by
epoch, but also removes the drift a real frequency change would produce.
Results are reported both ways and the truth lies between them.

**No type persists beyond about 2.5 days.** Persistent Tasman Sea blocking,
which lasts five to ten days, is split across types rather than isolated. If
southeast Australian rainfall and heat are driven substantially by blocking,
this stratification does not see that part.

**Frequencies are shares.** They sum to one, so one type cannot become more
common without others becoming less so. Individual frequency changes are a
redistribution, and types that are positive and negative phases of one pattern
are especially tied together.

---

## Reproducing all of it

```
python scripts/fetch_era5.py                       # MSLP via ARCO
python scripts/fetch_z500.py                       # Z500 via CDS
python scripts/fetch_silo.py --variables daily_rain max_temp

python scripts/preprocess.py                       # detrended
python scripts/preprocess.py --no-detrend --tag _nd

for tag in "" "_nd"; do
  python scripts/eof.py --tag "$tag"
  python scripts/classify.py --k 8 --modes 8 --tag "$tag"
  python scripts/composite.py --tag "$tag"
done

for index in rain tmax hot; do
  python scripts/attribute.py --index "$index" --seeds 5
  python scripts/attribute.py --index "$index" --tag _nd --seeds 5
done

python scripts/sam.py --tag _nd
python scripts/figures.py
```
