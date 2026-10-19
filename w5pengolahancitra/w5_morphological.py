import numpy as np
import matplotlib.pyplot as plt

def dilation(image, se):
    """Implementasi Dilation menggunakan 3x3 structuring element (se)."""
    img_h, img_w = image.shape
    se_h, se_w = se.shape
    pad_h = se_h // 2
    pad_w = se_w // 2
    
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image)
    
    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i:i+se_h, j:j+se_w]
            if np.any((se == 1) & (region == 1)):
                output[i, j] = 1
                
    return output

def erosion(image, se):
    """Implementasi Erosion menggunakan 3x3 structuring element (se)."""
    img_h, img_w = image.shape
    se_h, se_w = se.shape
    pad_h = se_h // 2
    pad_w = se_w // 2
    
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image)
    
    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i:i+se_h, j:j+se_w]
            mask = (se != -1)
            if np.all(region[mask] == se[mask]):
                output[i, j] = 1
                
    return output

def opening(image, se):
    """Implementasi Opening: Erosion diikuti Dilation."""
    return dilation(erosion(image, se), se)

def closing(image, se):
    """Implementasi Closing: Dilation diikuti Erosion."""
    return erosion(dilation(image, se), se)

def hit_or_miss_single_se(image, se):
    """
    Hit-or-Miss transform menggunakan 1 Structuring Element (SE) tunggal:
    1  = Foreground (harus cocok dengan nilai 1 di citra)
    0  = Background (harus cocok dengan nilai 0 di citra)
    -1 = Don't care (diabaikan)
    """
    img_h, img_w = image.shape
    se_h, se_w = se.shape
    pad_h = se_h // 2
    pad_w = se_w // 2
    
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image)
    
    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i:i+se_h, j:j+se_w]
            mask = (se != -1)
            if np.all(region[mask] == se[mask]):
                output[i, j] = 1
                
    return output

# Fungsi pembantu untuk mengatur grid gambar 10x10
def setup_image_grid(ax, img, title):
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which='minor', color='red', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks(range(0, 10, 2))
    ax.set_yticks(range(0, 10, 2))

# Fungsi pembantu untuk plot kernel agar tidak terpotong
def setup_kernel_grid(ax, k, title):
    ax.imshow(k, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=12)
    for i in range(3):
        for j in range(3):
            val_str = str(k[i, j])
            ax.text(j, i, val_str, ha='center', va='center', color='black', fontweight='bold', fontsize=14)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='--', linewidth=0.8)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)

# Matriks Citra Uji 10x10
sample_image = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Kernels
se_3x3 = np.ones((3, 3), dtype=int)
se_hom_single = np.array([
    [0, 1, 1],
    [0, 1, 1],
    [0, 0, 0]
])

def menu_dilasi_erosi():
    print("\n--- Menjalankan Dilation dan Erosion ---")
    res_dil = dilation(sample_image, se_3x3)
    res_ero = erosion(sample_image, se_3x3)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    setup_image_grid(axes[0, 0], sample_image, "Citra Asli")
    setup_image_grid(axes[0, 1], res_dil, "Hasil Dilation")
    setup_image_grid(axes[1, 0], res_ero, "Hasil Erosion")
    setup_kernel_grid(axes[1, 1], se_3x3, "Kernel SE 3x3\n(Dilation & Erosion)")
    
    filename = "output_dilasi_erosi.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Hasil visualisasi disimpan sebagai: {filename}")

def menu_opening_closing():
    print("\n--- Menjalankan Opening dan Closing ---")
    res_open = opening(sample_image, se_3x3)
    res_close = closing(sample_image, se_3x3)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    setup_image_grid(axes[0, 0], sample_image, "Citra Asli")
    setup_image_grid(axes[0, 1], res_open, "Hasil Opening")
    setup_image_grid(axes[1, 0], res_close, "Hasil Closing")
    setup_kernel_grid(axes[1, 1], se_3x3, "Kernel SE 3x3\n(Opening & Closing)")
    
    filename = "output_opening_closing.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Hasil visualisasi disimpan sebagai: {filename}")

def menu_hit_or_miss():
    print("\n--- Menjalankan Hit-or-Miss Transform ---")
    res_hom = hit_or_miss_single_se(sample_image, se_hom_single)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    plt.subplots_adjust(wspace=0.3)
    
    setup_image_grid(axes[0], sample_image, "Citra Asli")
    setup_image_grid(axes[1], res_hom, "Hasil Hit-or-Miss (Single SE)")
    setup_kernel_grid(axes[2], se_hom_single, "Kernel Tunggal Hit-or-Miss\n(1=Objek, 0=Background, -1=Don't Care)")
    
    filename = "output_hit_or_miss.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Hasil visualisasi disimpan sebagai: {filename}")

def main():
    while True:
        print("\n==========================================")
        print("    MENU INTERAKTIF OPERASI MORFOLOGI    ")
        print("==========================================")
        print("1. Dilasi dan Erosi")
        print("2. Opening dan Closing")
        print("3. Hit and Miss Transform")
        print("4. Keluar")
        print("==========================================")
        
        pilihan = input("Pilih menu (1-4): ").strip()
        
        if pilihan == '1':
            menu_dilasi_erosi()
        elif pilihan == '2':
            menu_opening_closing()
        elif pilihan == '3':
            menu_hit_or_miss()
        elif pilihan == '4':
            print("\nTerima kasih! Program selesai.")
            break
        else:
            print("\nPilihan tidak valid. Silakan masukkan angka 1, 2, 3, atau 4.")

if __name__ == "__main__":
    main()