import chromadb

# 1. Hubungkan ke database yang baru saja dibuat
client = chromadb.PersistentClient(path="./data/chroma_db")

# 2. Ambil koleksi yang sudah dibuat
collection = client.get_collection(name="toba_knowledge_base")

# 3. Hitung jumlah baris data yang masuk
jumlah_data = collection.count()
print(f"Jumlah dokumen di database: {jumlah_data}")

# 4. Cek konfigurasi ruang metrik HNSW (memastikan menggunakan cosine)
space = None
if hasattr(collection, "configuration") and collection.configuration:
    hnsw_cfg = collection.configuration.get("hnsw")
    if isinstance(hnsw_cfg, dict):
        space = hnsw_cfg.get("space")
if not space and collection.metadata:
    space = collection.metadata.get("hnsw:space")

print(f"Ruang jarak metrik HNSW: {space}")

if jumlah_data == 3308 and space == "cosine":
    print("✅ Berhasil! Semua data (3308) sudah masuk dengan metrik ruang 'cosine'.")
elif jumlah_data == 3308:
    print(f"✅ Dokumen lengkap (3308), ruang jarak: {space}")
else:
    print(f"❌ Ada yang kurang, datanya hanya {jumlah_data} dari 3308.")
