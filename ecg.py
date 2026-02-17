import wfdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from tensorflow.keras import layers, models, utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Cấu hình dữ liệu
RECORDS = ['100', '101', '103', '105', '111', '113', '117', '121']
WINDOW_SIZE = 180
HALF_WINDOW = 90

def load_data(records):
    all_beats = []
    all_labels = []
    
    print(">>> Tải dữ liệu từ PhysioNet...")
    for res in records:
        try:
            record = wfdb.rdrecord(res, pn_dir='mitdb')
            ann = wfdb.rdann(res, 'atr', pn_dir='mitdb')
            signal = record.p_signal[:, 0]
            
            # Chuẩn hóa tín hiệu
            signal = (signal - np.mean(signal)) / np.std(signal)
            
            for i in range(len(ann.sample)):
                pos = ann.sample[i]
                symbol = ann.symbol[i]
                
                if pos > HALF_WINDOW and pos < len(signal) - HALF_WINDOW:
                    beat = signal[pos - HALF_WINDOW : pos + HALF_WINDOW]
                    
                    # Gán nhãn: 0 (Thường), 1 (Bệnh)
                    if symbol == 'N':
                        all_beats.append(beat)
                        all_labels.append(0)
                    elif symbol in ['L', 'R', 'A', 'V', '/']:
                        all_beats.append(beat)
                        all_labels.append(1)
        except Exception as e:
            print(f"Lỗi bản ghi {res}: {e}")
            continue
            
    return np.array(all_beats), np.array(all_labels)

def main():
    # 1. Tiền xử lý
    X, y = load_data(RECORDS)
    X = X.reshape(-1, WINDOW_SIZE, 1)
    y_cat = utils.to_categorical(y, 2)

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    # 2. Định nghĩa kiến trúc ECG-Net
    model = models.Sequential([
        layers.Input(shape=(WINDOW_SIZE, 1)),
        layers.Conv1D(8, 5, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Conv1D(16, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # 3. Huấn luyện
    print("\n>>> Đang huấn luyện...")
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

    # 4. Đánh giá kết quả
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nĐộ chính xác: {acc*100:.2f}%")

    # 5. Lưu model và vẽ biểu đồ
    model.save('ecg_model.keras')
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('ECG Classification Result')
    plt.show()

if __name__ == "__main__":
    main()