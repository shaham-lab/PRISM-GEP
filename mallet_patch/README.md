# MALLET patch for PRISM-GEP

PRISM-GEP runs LDA with [Apache MALLET](https://mimno.github.io/Mallet/) 2.0.8, but with a
**modified topic-model core**. The Gibbs sampler is changed to accept an *informative
per-word* Dirichlet prior supplied on the command line as `--beta-file <csv>`. That informed
prior is Stage E of the method (see the top-level `README.md`) and is what distinguishes
PRISM-GEP's LDA from stock LDA.

Because of this, the build is **not** "stock MALLET 2.0.8 plus a one-line fix". It is a small
eight-file fork of MALLET's topic-model package. The modifications originate in the
**shaham-lab/PRISM** repository, under `beta_vector_patch/mallet/topics/`:

> https://github.com/shaham-lab/PRISM

All eight modified files are shipped in this directory, so no separate fork checkout is needed.

## The eight modified source files

Turning the scalar `beta` into a length-`V` vector changes a field type (`double beta` ->
`double[] beta`) and two constructor signatures (`WorkerRunnable`, `MarginalProbEstimator`,
`TopicInferencer`) that are referenced across the topic-model package. Every file that touches
the vector had to change with it, or `ant jar` fails with a type error. All eight live under
`cc/mallet/topics/` in a MALLET source tree and must replace their stock counterparts.

| Destination under `<mallet>/src/` | Why it changed |
|---|---|
| `cc/mallet/topics/ParallelTopicModel.java` | vector-`beta` prior + `--beta-file` loader (`loadBetaPrior`, `betaPath` field, `LabelAlphabet, double, String` constructor), plus the two bug fixes below. `beta` becomes `double[]`. |
| `cc/mallet/topics/WorkerRunnable.java` | its constructor and `resetBeta` take `double[] beta, double betaSum` instead of a scalar `double beta`; `ParallelTopicModel` calls both with the vector |
| `cc/mallet/topics/TopicInferencer.java` | field and constructor take `double[] beta`; built by `ParallelTopicModel.getInferencer()` with the vector |
| `cc/mallet/topics/MarginalProbEstimator.java` | constructor takes `double[] beta` plus `numTypes`/`betaSum`; built by `ParallelTopicModel.getProbEstimator()` with the vector |
| `cc/mallet/topics/PolylingualTopicModel.java` | passes its per-language `betas` array into the now-`double[]` `TopicInferencer` constructor |
| `cc/mallet/topics/TopicModelDiagnostics.java` | reads `model.beta` as `model.beta[0]` because `ParallelTopicModel.beta` is now a vector |
| `cc/mallet/topics/WeightedTopicModel.java` | `getEstimator()` calls the new `double[] beta` `MarginalProbEstimator` constructor |
| `cc/mallet/topics/tui/TopicTrainer.java` | defines the `--beta-file` command-line option and passes it into `ParallelTopicModel` |

`ParallelTopicModel.java` passes a `double[] beta` into `new WorkerRunnable(...)`, `resetBeta(...)`,
`new TopicInferencer(...)` and `new MarginalProbEstimator(...)`, so it does not compile against
the stock versions of those classes (which take a scalar `double beta`). `PolylingualTopicModel`,
`TopicModelDiagnostics` and `WeightedTopicModel` in turn reference those changed classes and the
now-vector `beta` field, so their stock versions no longer compile either. And a stock
`TopicTrainer` has no `--beta-file` option, so the informed prior never reaches the model. All
eight files are therefore required together — shipping fewer leaves `ant jar` with a compile error.

Do **not** bulk-copy the fork's `topics/` directory. It also carries unused scratch variants
(`WorkerRunnable1.java`, `WorkingWorkerRunnable.java`, `last_version_WorkerRunnable.java`) that
are not part of the build and are referenced by nothing. Copy only the eight files above (which
are exactly what this directory ships).

## What changed vs stock MALLET 2.0.8

1. **Informative per-word `beta` prior (the PRISM-GEP feature).**
   Stock MALLET takes a single scalar `beta`. The fork makes `beta` a length-`V` vector loaded
   from `--beta-file` (`ParallelTopicModel.loadBetaPrior`), threads it through `WorkerRunnable`,
   `TopicInferencer` and `MarginalProbEstimator`, and adds the `--beta-file` option to
   `TopicTrainer`. This is how the pipeline hands its Dirichlet `beta` prior CSV to LDA. MALLET
   reads the file positionally by type id, i.e. in first-encounter (data-CSV column) order, which
   is why the prior must be aligned to that order (see `scripts/train_prism_standard.py`).

2. **`printTopicWordWeights` printed an array reference and mutated `beta`.**
   The unpatched code did the equivalent of `double[] weight = beta; weight[topic] += ...;
   out.println(... + weight)`, which (a) wrote the Java array reference (e.g. `[D@387c703b`)
   into the topic-word-weights file instead of the numeric weight, and (b) mutated the shared
   `beta` array as a side effect. PRISM-GEP reads this file to extract per-topic gene weights,
   so the bug made the output unusable. The fix computes and prints the correct scalar weight
   (`double weight = beta[type]`) without touching `beta`.

3. **`optimizeBeta` allocated a dense `int[numTypes][maxTopicSize + 1]` and OOM'd.**
   For `V = 5000` types and a large `maxTopicSize`, the dense count histogram is on the order of
   ~4 GB and exhausts memory whenever `--optimize-interval` is on. The fix replaces the dense
   rectangular allocation with a **ragged** `int[numTypes][]` where each row is sized to that
   type's own maximum count (`int[maxCountForType + 1]`, typically far smaller). This drops the
   allocation to a few MB. `Dirichlet.learnParameters` iterates each row independently, so the
   ragged shape is behaviourally identical.

Fixes 2 and 3 are upstreamable and do not change the model's semantics. Feature 1 does. It is
the point of PRISM-GEP, and it is why the change spans eight files rather than one.

## How to build

You still need a stock **MALLET 2.0.8** checkout, but only for its Ant build file (`build.xml`),
its dependency jars, and the unmodified rest of `cc/mallet/`. The eight topic-model sources
below all come from this directory.

```bash
# 1. Get the MALLET 2.0.8 source tree (https://mimno.github.io/Mallet/) -> <mallet>/

# 2. Overlay the eight modified files onto the stock source tree:
cp mallet_patch/ParallelTopicModel.java     <mallet>/src/cc/mallet/topics/
cp mallet_patch/WorkerRunnable.java          <mallet>/src/cc/mallet/topics/
cp mallet_patch/TopicInferencer.java         <mallet>/src/cc/mallet/topics/
cp mallet_patch/MarginalProbEstimator.java   <mallet>/src/cc/mallet/topics/
cp mallet_patch/PolylingualTopicModel.java   <mallet>/src/cc/mallet/topics/
cp mallet_patch/TopicModelDiagnostics.java   <mallet>/src/cc/mallet/topics/
cp mallet_patch/WeightedTopicModel.java      <mallet>/src/cc/mallet/topics/
cp mallet_patch/tui/TopicTrainer.java        <mallet>/src/cc/mallet/topics/tui/

# 3. Pin the compiler encoding to UTF-8 (see the note below), then build:
cd <mallet> && ant -Dbuild.compiler.encoding=UTF-8 jar
```

**Pin the source encoding to UTF-8.** Two of the shipped files, `ParallelTopicModel.java`
and `WorkerRunnable.java`, carry UTF-8 characters in their comments (the Greek alpha and
beta letters, and the summation and proportionality symbols in the Gibbs update). Stock
MALLET 2.0.8 declares its `<javac>` task in `build.xml` without an `encoding` attribute, so
Ant falls back to the platform default charset. That is harmless where the default is
UTF-8, but on a machine whose default resolves to US-ASCII (a `LANG=C` or `POSIX` locale,
which is the usual case inside a bare Docker image or a CI runner) `javac` rejects those
bytes with `error: unmappable character for encoding US-ASCII` and emits no class files.
If your Ant version does not honour the property above, set the attribute directly on the
`<javac>` task in `<mallet>/build.xml`:

```xml
<javac
  source="${java_version}"
  target="${java_version}"
  encoding="UTF-8"
  destdir="${class}"
  ...
```

Place (or symlink) the built tree at `mallet/` in this repo so the classpath
`mallet/class` + `mallet/lib/mallet-deps.jar` resolves. That is the classpath
`scripts/train_prism_standard.py` and `prism_lib/ldamallet.py` use to invoke
`cc.mallet.topics.tui.TopicTrainer --beta-file`.

**Sanity check.** A correct build recognizes `--beta-file`. A stock build rejects it with
`Unrecognized option: --beta-file`. On POSIX (use `;` instead of `:` for the classpath on
Windows):

```bash
java -cp mallet/class:mallet/lib/mallet-deps.jar cc.mallet.topics.tui.TopicTrainer --help 2>&1 | grep beta-file
```

## `ParallelTopicModel.patch` (legacy, not a build route)

The `ParallelTopicModel.patch` file in this directory is a unified diff kept for reference only.
Its header labels the base as "stock MALLET 2.0.8", but its context lines already contain
fork-only symbols (`beta[]`, `betaPath`, `--beta-file`). It is in fact a diff between two fork
variants and will **not** apply to a stock 2.0.8 checkout. It also covers only
`ParallelTopicModel.java`, not the other six files. Use the file-overlay steps above. Do not run
`patch -p1` against stock MALLET.

## Attribution

The eight modified files (`ParallelTopicModel.java`, `WorkerRunnable.java`, `TopicInferencer.java`,
`MarginalProbEstimator.java`, `PolylingualTopicModel.java`, `TopicModelDiagnostics.java`,
`WeightedTopicModel.java` and `tui/TopicTrainer.java`) derive from Apache MALLET and remain under
MALLET's license (Common Public License 1.0). The PRISM-GEP modifications (the vector-`beta`
`--beta-file` prior and the two fixes above) are maintained in shaham-lab/PRISM.
