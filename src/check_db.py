import chromadb

# 1. Hubungkan ke database yang baru saja dibuat
client = chromadb.PersistentClient(path="./data/chroma_db")

# 2. Ambil koleksi yang sudah dibuat
collection = client.get_collection(name="toba_knowledge_base")

# 3. Hitung jumlah baris data yang masuk
jumlah_data = collection.count()
print(f"Jumlah dokumen di database: {jumlah_data}")

# 4. Cek konfigurasi metadata (memastikan menggunakan cosine)
print(f"Konfigurasi koleksi: {collection.metadata}")

if jumlah_data == 3308:
    print("✅ Berhasil! Semua data sudah masuk.")
else:
    print(f"❌ Ada yang kurang, datanya hanya {jumlah_data} dari 3308.")