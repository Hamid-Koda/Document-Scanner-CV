import matplotlib.pyplot as plt

def plot_all_learning_curves():

    def draw_subplots(epochs, train_no, val_no, train_drop, val_drop, title_base, y_label, filename, xticks_step):
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        axs[0].plot(epochs, train_no, label='Train Loss', color='#1f77b4', linewidth=2)
        axs[0].plot(epochs, val_no, label='Validation Loss', color='#ff7f0e', linewidth=2)
        axs[0].set_title(f'1. {title_base} (No Dropout)', fontsize=14, fontweight='bold')
        axs[0].set_xlabel('Epochs', fontsize=12)
        axs[0].set_ylabel(y_label, fontsize=12)
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].legend(fontsize=12)
        axs[0].set_xticks(range(0, max(epochs)+1, xticks_step))

        # با Dropout
        axs[1].plot(epochs, train_drop, label='Train Loss', color='#1f77b4', linewidth=2)
        axs[1].plot(epochs, val_drop, label='Validation Loss', color='#ff7f0e', linewidth=2)
        axs[1].set_title(f'2. {title_base} (With Dropout)', fontsize=14, fontweight='bold')
        axs[1].set_xlabel('Epochs', fontsize=12)
        axs[1].set_ylabel(y_label, fontsize=12)
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend(fontsize=12)
        axs[1].set_xticks(range(0, max(epochs)+1, xticks_step))

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f" The plot saved :  {filename}")
        plt.show()

    # ==============================================================
    # 1. Enhancement (20 Epochs)
    # ==============================================================
    enh_epochs = range(1, 21)
    enh_train_no = [0.3804, 0.3080, 0.2628, 0.2221, 0.1887, 0.1605, 0.1362, 0.1192, 0.1036, 0.0912, 
                    0.0807, 0.0723, 0.0652, 0.0591, 0.0564, 0.0522, 0.0500, 0.0464, 0.0450, 0.0436]
    enh_val_no =   [0.3389, 0.2880, 0.2450, 0.2096, 0.2282, 0.1470, 0.1367, 0.1111, 0.0960, 0.0824, 
                    0.0726, 0.0651, 0.0602, 0.0545, 0.0518, 0.0453, 0.0434, 0.0418, 0.0390, 0.0380]
    
    enh_train_drop = [0.4640, 0.4014, 0.3602, 0.3213, 0.2826, 0.2426, 0.2096, 0.1843, 0.1646, 0.1398, 
                      0.1198, 0.1059, 0.0943, 0.0844, 0.0768, 0.0716, 0.0646, 0.0597, 0.0561, 0.0536]
    enh_val_drop =   [0.4172, 0.3763, 0.3373, 0.3953, 0.2589, 0.2202, 0.1966, 0.1751, 0.1491, 0.1247, 
                      0.1111, 0.0974, 0.0849, 0.0746, 0.0687, 0.0627, 0.0578, 0.0542, 0.0489, 0.0465]

    print("Drawing Enhancement Model Curves...")
    draw_subplots(enh_epochs, enh_train_no, enh_val_no, enh_train_drop, enh_val_drop, 
                  "Enhancement U-Net", "Loss (EdgeAware)", "1_enhancement_curves.png", 2)

    # ==============================================================
    # 2. Direct Corner (20 Epochs)
    # ==============================================================
    dir_epochs = range(1, 21)
    dir_train_no = [0.0121, 0.0043, 0.0032, 0.0024, 0.0019, 0.0016, 0.0013, 0.0011, 0.0010, 0.0009, 
                    0.0008, 0.0008, 0.0007, 0.0006, 0.0006, 0.0006, 0.0005, 0.0005, 0.0005, 0.0004]
    dir_val_no =   [0.0057, 0.0047, 0.0032, 0.0026, 0.0022, 0.0019, 0.0017, 0.0014, 0.0012, 0.0009, 
                    0.0008, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0004, 0.0004, 0.0004]
                         
    dir_train_drop = [0.0151, 0.0072, 0.0062, 0.0055, 0.0049, 0.0044, 0.0040, 0.0036, 0.0034, 0.0032, 
                      0.0030, 0.0029, 0.0028, 0.0027, 0.0026, 0.0024, 0.0025, 0.0023, 0.0022, 0.0022]
    dir_val_drop =   [0.0091, 0.0066, 0.0055, 0.0052, 0.0041, 0.0036, 0.0030, 0.0025, 0.0027, 0.0023, 
                      0.0023, 0.0020, 0.0021, 0.0018, 0.0022, 0.0020, 0.0018, 0.0017, 0.0016, 0.0015]

    print("Drawing Direct Corner Model Curves...")
    draw_subplots(dir_epochs, dir_train_no, dir_val_no, dir_train_drop, dir_val_drop, 
                  "Direct Corner Regressor", "Loss", "2_direct_corner_curves.png", 2)

    # ==============================================================
    # 3. Heatmap Corner (25 Epochs)
    # ==============================================================
    hm_epochs = range(1, 26)
    hm_train_no = [0.5248, 0.4299, 0.3907, 0.3597, 0.3320, 0.3068, 0.2823, 0.2602, 0.2428, 0.2238, 
                   0.2076, 0.1935, 0.1801, 0.1675, 0.1561, 0.1460, 0.1366, 0.1274, 0.1181, 0.1096, 
                   0.1021, 0.0957, 0.0905, 0.0854, 0.0805]
    hm_val_no =   [0.4465, 0.4113, 0.3686, 0.3419, 0.3189, 0.2949, 0.2712, 0.2486, 0.2322, 0.2131, 
                   0.1962, 0.1830, 0.1700, 0.1680, 0.1511, 0.1407, 0.1323, 0.1251, 0.1194, 0.1038, 
                   0.0986, 0.0930, 0.0871, 0.0838, 0.0783]
                        
    hm_train_drop = [0.5766, 0.4800, 0.4400, 0.4072, 0.3748, 0.3475, 0.3189, 0.2912, 0.2663, 0.2448, 
                     0.2259, 0.2093, 0.1943, 0.1805, 0.1682, 0.1569, 0.1466, 0.1380, 0.1305, 0.1224, 
                     0.1163, 0.1095, 0.1026, 0.0941, 0.0872]
    hm_val_drop =   [0.4931, 0.4530, 0.4221, 0.3913, 0.3584, 0.3326, 0.2993, 0.2784, 0.2592, 0.2390, 
                     0.2192, 0.2060, 0.1885, 0.1741, 0.1616, 0.1537, 0.1445, 0.1383, 0.1272, 0.1186, 
                     0.1164, 0.1049, 0.1000, 0.0939, 0.0858]

    print("Drawing Heatmap Corner Model Curves...")
    draw_subplots(hm_epochs, hm_train_no, hm_val_no, hm_train_drop, hm_val_drop, 
                  "Heatmap Corner Regressor", "Loss", "3_heatmap_corner_curves.png", 5)

if __name__ == '__main__':
    print(" Starting curve generation process...")
    plot_all_learning_curves()
    print(" All 3 charts generated successfully!")