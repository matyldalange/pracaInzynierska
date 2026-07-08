import os
import shutil

nowy_folder_bez_masek = '/Users/matyldalange/Desktop/ZBIOR-koniec/100a/'
nowy_folder_z_maskami = '/Users/matyldalange/Desktop/ZBIOR-koniec/100maskia/'

test_folder_bez_masek = '/Users/matyldalange/Desktop/ZBIOR-koniec/100testa/'
test_folder_z_maskami = '/Users/matyldalange/Desktop/ZBIOR-koniec/100maskitesta/'
val_folder_bez_masek = '/Users/matyldalange/Desktop/ZBIOR-koniec/100vala/'
val_folder_z_maskami = '/Users/matyldalange/Desktop/ZBIOR-koniec/100maskivala/'
train_folder_bez_masek = '/Users/matyldalange/Desktop/ZBIOR-koniec/100traina/'
train_folder_z_maskami = '/Users/matyldalange/Desktop/ZBIOR-koniec/100maskitraina/'

os.makedirs(test_folder_bez_masek, exist_ok=True)
os.makedirs(test_folder_z_maskami, exist_ok=True)
os.makedirs(val_folder_bez_masek, exist_ok=True)
os.makedirs(val_folder_z_maskami, exist_ok=True)
os.makedirs(train_folder_bez_masek, exist_ok=True)
os.makedirs(train_folder_z_maskami, exist_ok=True)

pliki_bez_masek = os.listdir(nowy_folder_bez_masek)
pliki_z_maskami = os.listdir(nowy_folder_z_maskami)

#Parujemy pliki na podstawie pierwszych 6 cyfr w nazwie
pary_plikow = []
for plik_bez_maski in pliki_bez_masek:
    identyfikator = plik_bez_maski[:6]
    odpowiadajacy_plik_z_maska = next((plik for plik in pliki_z_maskami if plik.startswith(identyfikator)), None)
    if odpowiadajacy_plik_z_maska:
        pary_plikow.append((plik_bez_maski, odpowiadajacy_plik_z_maska))

# Podział zbioru na przykładowe proporcje: 10% obrazy testowe, 10% walidacyjne, 80% treningowe
n = len(pary_plikow)
test_pary = pary_plikow[:n // 10]
val_pary = pary_plikow[n // 10: 2 * n // 10]
train_pary = pary_plikow[2 * n // 10:]


# Funkcja kopiująca pary plików
def copy_pairs(pairs, source_folder_bez_maski, source_folder_z_maskami, dest_folder_bez_maski, dest_folder_z_maskami):
    for plik_bez_maski, plik_z_maska in pairs:
        shutil.copy(os.path.join(source_folder_bez_maski, plik_bez_maski), os.path.join(dest_folder_bez_maski, plik_bez_maski))
        shutil.copy(os.path.join(source_folder_z_maskami, plik_z_maska), os.path.join(dest_folder_z_maskami, plik_z_maska))

copy_pairs(test_pary, nowy_folder_bez_masek, nowy_folder_z_maskami, test_folder_bez_masek, test_folder_z_maskami)
copy_pairs(val_pary, nowy_folder_bez_masek, nowy_folder_z_maskami, val_folder_bez_masek, val_folder_z_maskami)
copy_pairs(train_pary, nowy_folder_bez_masek, nowy_folder_z_maskami, train_folder_bez_masek, train_folder_z_maskami)

