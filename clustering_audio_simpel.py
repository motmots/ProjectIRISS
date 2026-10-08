"""
Clustering Audio (versi simpel, TANPA model pretrained)
Alur: baca audio -> preprocessing -> ekstraksi fitur klasik (MFCC dkk) -> EDA
      -> scaling + PCA -> coba beberapa k & algoritma -> pilih silhouette terbaik -> analisis tiap cluster

Install: pip install librosa soundfile scikit-learn matplotlib pandas
(file mp3/m4a butuh ffmpeg terpasang di komputer)
"""
import os
import glob
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, silhouette_samples, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score)

try:                                   # untuk memutar audio hasil cluster, hanya jalan di notebook
    from IPython import get_ipython
    from IPython.display import Audio, display
    DI_NOTEBOOK = get_ipython() is not None
except ImportError:
    DI_NOTEBOOK = False

# ============ PENGATURAN (ubah sesuai data) ============
FOLDER = "dataset_audio"           # folder audio (subfolder ikut dibaca)
SR = 22050                         # semua audio disamakan ke sample rate ini
DURASI_MAKS = 30                   # hanya baca N detik pertama (None = baca semua)
TRIM_HENING = True                 # buang hening di awal/akhir
TOP_DB = 30                        # makin kecil makin agresif membuang hening
NORMALISASI = True                 # samakan volume puncak tiap audio
MIN_DURASI = 1.0                   # buang audio < N detik (setelah trim)
MIN_RMS = 0.001                    # buang audio hampir hening total. 0 = nonaktif
BUANG_DUPLIKAT = True              # buang file duplikat persis
N_MFCC = 20                        # jumlah koefisien MFCC

PAKAI_SCALER = True                # StandardScaler sebelum PCA (disarankan, skala fitur beda-beda)
DAFTAR_PCA = [10, 20]              # jumlah komponen PCA yang dicoba
DAFTAR_ALGO = ["kmeans", "agglo"]  # kmeans | agglo | gmm
DAFTAR_K = range(2, 11)            # k yang dicoba (kecilkan kalau data < 50)
K_MANUAL = None                    # None = otomatis dari silhouette, atau isi angka
MIN_UKURAN = 0.02                  # cluster < 2% data tidak boleh jadi pilihan terbaik
JUMLAH_CONTOH = 3                  # contoh audio per cluster
MAKS_SIL = 5000                    # data > ini: silhouette dihitung dari sampel
SEED = 42
# ========================================================

np.random.seed(SEED)

# ---------- 1. Baca path audio ----------
ekstensi = ("wav", "mp3", "flac", "ogg", "m4a")
semua_path = sorted(set(p for e in ekstensi + tuple(x.upper() for x in ekstensi)
                        for p in glob.glob(os.path.join(FOLDER, "**", f"*.{e}"), recursive=True)))
print("Jumlah file audio:", len(semua_path))
assert len(semua_path) > 10, "Audio tidak ketemu / terlalu sedikit, cek FOLDER"


# ---------- 2. PREPROCESSING + EKSTRAKSI FITUR ----------
def praproses(path):
    """Baca audio lalu rapikan. Return (sinyal, alasan_dibuang, rms_awal)."""
    y, _ = librosa.load(path, sr=SR, mono=True, duration=DURASI_MAKS)   # jadikan mono + samakan sample rate
    rms_awal = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 0.0
    if rms_awal < MIN_RMS:
        return None, "hening", rms_awal
    if TRIM_HENING:
        y, _ = librosa.effects.trim(y, top_db=TOP_DB)                  # buang hening awal/akhir
    if len(y) / SR < MIN_DURASI:
        return None, "terlalu_pendek", rms_awal
    if NORMALISASI:
        y = y / (np.max(np.abs(y)) + 1e-9)                             # volume puncak = 1
    return y, None, rms_awal


def ekstrak_fitur(y):
    """Satu audio -> satu vektor fitur (rata-rata & std terhadap waktu)."""
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR)
    kontras = librosa.feature.spectral_contrast(y=y, sr=SR)
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    skalar = [centroid, bandwidth, rolloff, zcr, rms]
    # urutan kolom: mfcc mean | mfcc std | delta mfcc mean | chroma (12) | spectral contrast (7) | 5 fitur x (mean, std)
    fitur = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), delta.mean(axis=1),
                            chroma.mean(axis=1), kontras.mean(axis=1)]
                           + [np.array([f.mean(), f.std()]) for f in skalar])
    ringkas = dict(centroid=centroid.mean(), bandwidth=bandwidth.mean(), rolloff=rolloff.mean(),
                   zcr=zcr.mean(), rms=rms.mean())
    return fitur, ringkas


dibuang = {"rusak": 0, "duplikat": 0, "hening": 0, "terlalu_pendek": 0}
hash_terlihat = set()
list_info, list_fitur = [], []
for i, p in enumerate(semua_path):
    if BUANG_DUPLIKAT:
        with open(p, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h in hash_terlihat:
            dibuang["duplikat"] += 1
            continue
        hash_terlihat.add(h)
    try:
        y, alasan, rms_awal = praproses(p)
        if alasan is not None:
            dibuang[alasan] += 1
            continue
        fitur, ringkas = ekstrak_fitur(y)
        sr_asli = librosa.get_samplerate(p)
        durasi_asli = librosa.get_duration(path=p)
    except Exception as e:
        print("Audio rusak dilewati:", p, "|", str(e)[:80])
        dibuang["rusak"] += 1
        continue
    list_info.append(dict(path=p, folder=os.path.basename(os.path.dirname(p)),
                          format=os.path.splitext(p)[1].lower(), sr_asli=sr_asli,
                          durasi_asli=durasi_asli, durasi=len(y) / SR, rms_awal=rms_awal,
                          ukuran_kb=os.path.getsize(p) / 1024, **ringkas))
    list_fitur.append(fitur)
    print(f"  diproses {i + 1}/{len(semua_path)}", end="\r")

data_eda = pd.DataFrame(list_info)          # satu baris = satu audio
X = np.array(list_fitur)
path_valid = list(data_eda["path"])
punya_label = 1 < data_eda["folder"].nunique() <= 50      # subfolder dianggap label

print("\nAudio dibuang saat preprocessing:", dibuang)
assert len(X) > 10, "Audio valid terlalu sedikit, cek MIN_DURASI / MIN_RMS / format file"
print("Audio valid:", len(X), "| bentuk fitur:", X.shape)


# ---------- 3. EDA ----------
print(data_eda[["durasi_asli", "durasi", "rms_awal", "centroid", "zcr", "ukuran_kb"]].describe().round(3))
print("\nFormat file:\n", data_eda["format"].value_counts())
print("\nSample rate asli:\n", data_eda["sr_asli"].value_counts())
if punya_label:
    print("\nJumlah per folder:\n", data_eda["folder"].value_counts())


def gambar_audio(path, ax_wave, ax_mel, judul=""):
    """Waveform + mel-spectrogram satu audio (sudah dipraproses)."""
    y, _, _ = praproses(path)
    librosa.display.waveshow(y, sr=SR, ax=ax_wave, color="steelblue")
    ax_wave.set_title(judul, fontsize=8); ax_wave.set_xlabel("")
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR), ref=np.max)
    librosa.display.specshow(mel, sr=SR, x_axis="time", y_axis="mel", ax=ax_mel, cmap="magma")
    ax_mel.set_ylabel("")


# contoh audio acak: waveform + mel-spectrogram
idx = np.random.RandomState(SEED).choice(len(path_valid), size=min(4, len(path_valid)), replace=False)
fig, axes = plt.subplots(2, len(idx), figsize=(4 * len(idx), 5.5), squeeze=False)
for j, i in enumerate(idx):
    nama = (data_eda["folder"][i] + "/" if punya_label else "") + os.path.basename(path_valid[i])
    gambar_audio(path_valid[i], axes[0, j], axes[1, j], nama[:30])
fig.suptitle("Contoh Audio: Waveform (atas) & Mel-Spectrogram (bawah)"); fig.tight_layout(); plt.show()

# distribusi properti
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for a, (kol, judul) in zip(axes.ravel(), [("durasi_asli", "Durasi asli (detik)"), ("durasi", "Durasi setelah trim (detik)"),
                                          ("rms_awal", "RMS (kekerasan suara)"), ("centroid", "Spectral centroid (Hz)"),
                                          ("zcr", "Zero-crossing rate"), ("bandwidth", "Spectral bandwidth (Hz)")]):
    a.hist(data_eda[kol], bins=30, color="steelblue"); a.set_title(judul)
fig.suptitle("Distribusi Properti Audio"); fig.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()

# sample rate & jumlah per folder
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sr_hitung = data_eda["sr_asli"].value_counts()
ax[0].bar(sr_hitung.index.astype(str), sr_hitung.values, color="tab:green"); ax[0].set(title="Sample Rate Asli", xlabel="Hz")
jumlah = data_eda["folder"].value_counts().head(20) if punya_label else data_eda["format"].value_counts()
ax[1].bar(jumlah.index.astype(str), jumlah.values, color="tab:orange"); ax[1].tick_params(axis="x", rotation=60)
ax[1].set_title("Jumlah per Folder" if punya_label else "Format File")
fig.tight_layout(); plt.show()


# ---------- 4. Scaling + PCA ----------
X_scaled = StandardScaler().fit_transform(X) if PAKAI_SCALER else X
data_pca = {}
for jml in DAFTAR_PCA:
    jml = min(jml, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=jml, random_state=SEED)
    data_pca[jml] = pca.fit_transform(X_scaled)
    print(f"PCA {jml} komponen -> variance dijelaskan {pca.explained_variance_ratio_.sum():.2%}")


# ---------- 5. Coba beberapa pengaturan ----------
def buat_model(algo, k):
    if algo == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=SEED)
    if algo == "agglo":
        return AgglomerativeClustering(n_clusters=k, linkage="ward")
    if algo == "gmm":
        return GaussianMixture(n_components=k, covariance_type="diag", random_state=SEED)
    raise ValueError("algo tidak dikenal: " + algo)


def hitung_sil(Xp, lab):
    sampel = MAKS_SIL if len(Xp) > MAKS_SIL else None
    return silhouette_score(Xp, lab, sample_size=sampel, random_state=SEED)


baris = []
for jml, Xp in data_pca.items():
    for algo in DAFTAR_ALGO:
        for k in DAFTAR_K:
            model = buat_model(algo, k)
            lab = model.fit_predict(Xp)
            if len(set(lab)) < 2:
                continue
            baris.append(dict(n_pca=jml, algo=algo, k=k, silhouette=hitung_sil(Xp, lab),
                              davies_bouldin=davies_bouldin_score(Xp, lab),
                              calinski=calinski_harabasz_score(Xp, lab),
                              inertia=getattr(model, "inertia_", np.nan),
                              cluster_terkecil=int(np.bincount(lab).min())))
hasil_percobaan = pd.DataFrame(baris)

print("\n=== 15 percobaan terbaik (urut silhouette) ===")
print(hasil_percobaan.sort_values("silhouette", ascending=False).head(15).round(4).to_string(index=False))
print("\n=== Terbaik untuk tiap (algo, n_pca) ===")
print(hasil_percobaan.loc[hasil_percobaan.groupby(["algo", "n_pca"]).silhouette.idxmax()].round(4).to_string(index=False))

kandidat = hasil_percobaan if K_MANUAL is None else hasil_percobaan[hasil_percobaan.k == K_MANUAL]
kandidat_aman = kandidat[kandidat.cluster_terkecil >= max(2, MIN_UKURAN * len(X))]
if len(kandidat_aman) > 0:
    kandidat = kandidat_aman
terbaik = kandidat.sort_values("silhouette", ascending=False).iloc[0]
algo_terbaik, k_terbaik, pca_terbaik = terbaik.algo, int(terbaik.k), int(terbaik.n_pca)
Xp = data_pca[pca_terbaik]
print(f"\n>>> Dipilih: algo={algo_terbaik}, n_pca={pca_terbaik}, k={k_terbaik} (silhouette={terbaik.silhouette:.4f})")


# ---------- 6. Plot elbow dan silhouette vs k ----------
def cari_elbow(ks, ys):
    """Titik yang paling jauh dari garis lurus ujung-ke-ujung kurva."""
    ks, ys = np.array(ks, float), np.array(ys, float)
    if len(ks) < 3 or ys.max() == ys.min():
        return int(ks[0])
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (ys - ys.min()) / (ys.max() - ys.min())
    return int(ks[np.abs(x + y - 1).argmax()])


if "kmeans" in DAFTAR_ALGO:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    print("\n=== Elbow (KMeans) ===")
    for jml in data_pca:
        d = hasil_percobaan[(hasil_percobaan.algo == "kmeans") & (hasil_percobaan.n_pca == jml)].sort_values("k")
        ax.plot(d.k, d.inertia, "o-", label=f"PCA {jml}")
        k_elbow = cari_elbow(d.k, d.inertia)
        ax.plot(k_elbow, d.inertia[d.k == k_elbow].values[0], "r*", ms=14)
        print(f"  n_pca={jml}: elbow di k = {k_elbow}")
    ax.set(title="Elbow Method (bintang merah = elbow otomatis)", xlabel="k", ylabel="Inertia")
    ax.legend(); fig.tight_layout(); plt.show()

fig, axes = plt.subplots(1, len(DAFTAR_ALGO), figsize=(6 * len(DAFTAR_ALGO), 4.3), squeeze=False)
for a, algo in zip(axes[0], DAFTAR_ALGO):
    for jml in data_pca:
        d = hasil_percobaan[(hasil_percobaan.algo == algo) & (hasil_percobaan.n_pca == jml)].sort_values("k")
        a.plot(d.k, d.silhouette, "o-", label=f"PCA {jml}")
    a.set(title=f"Silhouette vs k ({algo})", xlabel="k", ylabel="Silhouette"); a.legend()
fig.tight_layout(); plt.show()


# ---------- 7. Model final + evaluasi ----------
label = buat_model(algo_terbaik, k_terbaik).fit_predict(Xp)
sil_total = silhouette_score(Xp, label)
sil_sampel = silhouette_samples(Xp, label)
print(f"\nSilhouette Score  : {sil_total:.4f}")
print(f"Davies-Bouldin    : {davies_bouldin_score(Xp, label):.4f}  (makin kecil makin bagus)")
print(f"Calinski-Harabasz : {calinski_harabasz_score(Xp, label):.2f}  (makin besar makin bagus)")

data_eda["cluster"] = label
data_eda["silhouette"] = sil_sampel
daftar_cluster = sorted(set(label))
print("\nJumlah audio per cluster:\n", data_eda["cluster"].value_counts().sort_index())
print("\nRata-rata silhouette per cluster:\n", data_eda.groupby("cluster")["silhouette"].mean().round(4))

# plot cluster 2D
tsne = TSNE(n_components=2, perplexity=max(2, min(30, (len(Xp) - 1) / 3)),
            random_state=SEED, init="pca", learning_rate="auto").fit_transform(Xp)
pca2 = PCA(n_components=2, random_state=SEED).fit_transform(Xp)
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
for a, hasil_2d, nama in zip(ax, [tsne, pca2], ["t-SNE", "PCA"]):
    sc = a.scatter(hasil_2d[:, 0], hasil_2d[:, 1], c=label, cmap="tab10", s=18, alpha=.8)
    a.set(title=f"Plot Cluster ({nama}), {algo_terbaik}, k={k_terbaik}", xlabel="Dim 1", ylabel="Dim 2")
    a.legend(*sc.legend_elements(), title="Cluster")
fig.tight_layout(); plt.show()

# silhouette plot
fig, ax = plt.subplots(figsize=(8, 6))
bawah = 10
for c in daftar_cluster:
    s = np.sort(sil_sampel[label == c]); atas = bawah + len(s)
    ax.fill_betweenx(np.arange(bawah, atas), 0, s, alpha=.75, color=plt.cm.tab10(c % 10))
    ax.text(-0.05, bawah + .5 * len(s), str(c)); bawah = atas + 10
ax.axvline(sil_total, c="red", ls="--", label=f"rata-rata = {sil_total:.3f}")
ax.set(title="Silhouette Plot per Cluster", xlabel="Silhouette", ylabel="Cluster", yticks=[])
ax.legend(); fig.tight_layout(); plt.show()

# ukuran cluster
jumlah = data_eda["cluster"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(jumlah.index.astype(str), jumlah.values, color=[plt.cm.tab10(i % 10) for i in jumlah.index])
for i, v in enumerate(jumlah.values):
    ax.text(i, v, str(v), ha="center", va="bottom")
ax.set(title="Jumlah Audio per Cluster", xlabel="Cluster", ylabel="Jumlah")
fig.tight_layout(); plt.show()


# ---------- 8. Interpretasi tiap cluster ----------
print("\nRata-rata properti per cluster:\n",
      data_eda.groupby("cluster")[["durasi", "rms_awal", "centroid", "bandwidth", "rolloff", "zcr"]].mean().round(3))

# contoh audio paling dekat pusat cluster (mel-spectrogram), dan bisa diputar kalau di notebook
contoh = {}
for c in daftar_cluster:
    anggota = np.where(label == c)[0]
    pusat = Xp[anggota].mean(axis=0)
    contoh[c] = anggota[np.argsort(np.linalg.norm(Xp[anggota] - pusat, axis=1))[:JUMLAH_CONTOH]]

fig, axes = plt.subplots(len(daftar_cluster), JUMLAH_CONTOH,
                         figsize=(4 * JUMLAH_CONTOH, 2.4 * len(daftar_cluster)), squeeze=False)
for r, c in enumerate(daftar_cluster):
    for j in range(JUMLAH_CONTOH):
        a = axes[r, j]
        if j < len(contoh[c]):
            y, _, _ = praproses(path_valid[contoh[c][j]])
            mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR), ref=np.max)
            librosa.display.specshow(mel, sr=SR, x_axis="time", y_axis="mel", ax=a, cmap="magma")
            a.set_title(f"Cluster {c}: " + os.path.basename(path_valid[contoh[c][j]])[:22], fontsize=8)
        else:
            a.axis("off")
fig.suptitle("Mel-Spectrogram Contoh per Cluster (paling dekat pusat)"); fig.tight_layout(rect=[0, 0, 1, 0.97]); plt.show()

if DI_NOTEBOOK:
    for c in daftar_cluster:
        print(f"\nCluster {c}: putar contoh audio")
        for i in contoh[c]:
            y, _, _ = praproses(path_valid[i])
            print("  ", os.path.basename(path_valid[i]))
            display(Audio(y, rate=SR))

# properti per cluster
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for a, (kol, judul) in zip(axes.ravel(), [("durasi", "Durasi (detik)"), ("rms_awal", "RMS (kekerasan suara)"),
                                          ("centroid", "Spectral centroid (Hz)"), ("zcr", "Zero-crossing rate")]):
    a.boxplot([data_eda.loc[data_eda.cluster == c, kol] for c in daftar_cluster], showfliers=False)
    a.set_xticklabels([str(c) for c in daftar_cluster]); a.set(title=judul, xlabel="Cluster")
fig.suptitle("Properti Audio per Cluster"); fig.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()

# rata-rata MFCC per cluster (N_MFCC kolom pertama di X = rata-rata MFCC)
rata_mfcc = np.array([X[label == c, :N_MFCC].mean(axis=0) for c in daftar_cluster])
fig, ax = plt.subplots(figsize=(1.5 + 0.9 * len(daftar_cluster), 6))
im = ax.imshow(rata_mfcc.T, aspect="auto", cmap="coolwarm", origin="lower")
ax.set_xticks(range(len(daftar_cluster))); ax.set_xticklabels([f"C{c}" for c in daftar_cluster])
ax.set(ylabel="MFCC ke-", title="Rata-rata MFCC per Cluster"); fig.colorbar(im, ax=ax)
fig.tight_layout(); plt.show()

if punya_label:       # bermakna kalau nama folder = label asli
    tabel = pd.crosstab(data_eda.cluster, data_eda.folder)
    print("\nCluster vs folder:\n", tabel)
    print(f"ARI = {adjusted_rand_score(data_eda.folder, data_eda.cluster):.4f} | "
          f"NMI = {normalized_mutual_info_score(data_eda.folder, data_eda.cluster):.4f}  (1 = cocok sempurna)")
