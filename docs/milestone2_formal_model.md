# Formalisasi Matematis: Toba Copilot Business Constraint Solver (Milestone 2)

Dokumen ini mendeskripsikan model formal matematis untuk **Constraint Optimization Problem (COP)** yang diterapkan pada pemilihan paket wisata (Solver) di sistem Toba Copilot. Model ini disusun berdasarkan implementasi pada `solver.py`.

## 1. Definisi Variabel ($X$)

Terdapat himpunan variabel diskrit yang merepresentasikan kategori item dalam satu paket wisata:

$$X = \{x_1, x_2, x_3, x_4\}$$

di mana:
- $x_1$ = `homestay` (Akomodasi penginapan)
- $x_2$ = `kuliner` (Tempat makan)
- $x_3$ = `destinasi_utama` (Tempat wisata reguler)
- $x_4$ = `destinasi_umkm` (Tempat wisata berbasis UMKM)

Setiap kombinasi solusi (assignment) harus memiliki tepat satu nilai untuk masing-masing variabel tersebut.

## 2. Domain ($D$)

Setiap variabel $x_i \in X$ memiliki domain $D_i$ yang berisi daftar kandidat objek `ItemWisata` yang relevan dengan kategorinya.

Setiap elemen $v \in D_i$ merupakan sebuah *tuple* atribut (berdasarkan atribut dataclass `ItemWisata`):
$$v = \langle \text{id}, \text{nama}, \text{kategori}, \text{harga}, \text{lokasi}, \text{is\_umkm} \rangle$$

Dengan syarat bahwa untuk setiap $v \in D_i$, properti kategori harus konsisten dengan variabelnya:
$$v.\text{kategori} = x_i$$

## 3. Batasan / Constraints ($C$)

Tidak seperti CSP klasik yang sering didominasi batasan biner (antara 2 variabel), problem ini menggunakan **global constraints** yang melibatkan seluruh assignment variabel secara kolektif:

### a. Constraint Global Budget
Jumlah harga dari seluruh item wisata yang dipilih tidak boleh melebihi batas anggaran ($B$) wisatawan:
$$\sum_{i=1}^{4} v_i.\text{harga} \le B$$
di mana $v_i$ adalah nilai yang di-assign pada variabel $x_i$ dan $B$ adalah `budget`.

### b. Constraint Global UMKM
Jika aturan regulasi `require_umkm = True`, maka setidaknya satu dari item wisata yang dipilih di dalam keseluruhan paket harus merepresentasikan UMKM lokal:
$$\bigvee_{i=1}^{4} v_i.\text{is\_umkm} = \text{True}$$

## 4. Fungsi Objektif (Constraint Optimization Problem)

Karena tujuan dari solver ini tidak hanya menemukan solusi yang valid (satisfaction) melainkan mencari paket wisata dengan biaya *minimum* (termurah) yang masih layak (feasible), ini dikategorikan sebagai **Constraint Optimization Problem**.

Fungsi objektif yang ingin diminimalkan adalah total harga:
$$\min \sum_{i=1}^{4} v_i.\text{harga}$$

## 5. Strategi Algoritma Solver

Solver yang dikembangkan pada `solver.py` tidak murni menggunakan Constraint Satisfaction tradisional, melainkan menggabungkan beberapa strategi pencarian dan propagasi:

### a. Generalized Arc Consistency (GAC)
Karena batasan *budget* dan *UMKM* merupakan "global constraints" (melibatkan semua variabel $x_1, x_2, x_3, x_4$), maka algoritma AC-3 biner klasik tidak memadai. Kode ini mengadaptasi **Generalized Arc Consistency (GAC)**.

Hal ini secara eksplisit dijelaskan pada dokumentasi (komentar) kode:
> *"The budget and UMKM rules are global constraints rather than binary relations, so this applies generalized arc consistency (GAC), the appropriate extension of AC-3 for these domains."* (pada fungsi `apply_ac3_node_and_arc_consistency()`).

Fungsi `_propagate_global_constraints()` bertugas melakukan pemangkasan iteratif *(pruning)*. Setiap kandidat nilai pada suatu variabel akan dievaluasi apakah ia masih memiliki *support* (dukungan) biaya yang layak di variabel-variabel lain menggunakan perhitungan `_minimum_completion_cost`. Jika suatu kandidat akan secara absolut menyebabkan *over-budget*, kandidat tersebut dihapus dari domain.

### b. Minimum Remaining Values (MRV) Heuristic
Pada fase rekursi *Backtracking*, strategi seleksi variabel menggunakan heuristik **MRV (Minimum Remaining Values)** yang diimplementasikan pada fungsi `select_unassigned_variable_mrv()`. 

Algoritma akan memprioritaskan (memilih) variabel kosong yang memiliki ukuran domain ($|D_i|$) paling kecil (paling sedikit kandidat tersisanya). Heuristik ini bertujuan untuk mencapai kegagalan lebih awal (fail-first principle) guna memangkas cabang pencarian yang tidak valid dengan cepat.

### c. Branch and Bound Optimization
Untuk menangani elemen optimasi (Fungsi Objektif), *Backtracking* yang digunakan dikombinasikan dengan teknik **Branch and Bound**. 
- Solver melacak biaya solusi valid terbaik sejauh ini dalam `self._best_cost`.
- Pada setiap simpul di pohon pencarian, fungsi menghitung batas bawah (*lower bound*) penyelesaian rute:
  $$\text{lower\_bound} = \text{current\_cost} + \text{minimum\_completion\_cost}$$
- Jika `lower_bound >= self._best_cost`, maka cabang rekursif tersebut dihentikan (dipangkas/prune) karena tidak mungkin menghasilkan paket wisata yang lebih murah daripada solusi terbaik yang sudah ditemukan.
