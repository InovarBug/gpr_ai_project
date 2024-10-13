import os
import shutil
from tkinter import filedialog, simpledialog, Tk, messagebox, Toplevel, Label, Listbox, Button, MULTIPLE

def select_images(root):
    root.withdraw()  # Oculta a janela principal do Tkinter
    image_files = filedialog.askopenfilenames(
        title="Selecione as Imagens",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    return image_files

def select_dataset_dir(root):
    root.withdraw()
    dataset_dir = filedialog.askdirectory(title="Selecione o Diretório do Dataset")
    return dataset_dir

def show_images(image_files):
    preview_window = Toplevel()
    preview_window.title("Imagens Selecionadas")

    listbox = Listbox(preview_window, selectmode=MULTIPLE, width=50)
    listbox.pack(pady=10)

    for image_path in image_files:
        listbox.insert("end", os.path.basename(image_path))

    Label(preview_window, text="Imagens selecionadas:").pack(pady=5)
    Label(preview_window, text=f"Total: {len(image_files)} imagens").pack(pady=5)

    Button(preview_window, text="Fechar", command=preview_window.destroy).pack(pady=10)

    preview_window.mainloop()

def organize_images_into_dataset():
    root = Tk()
    image_files = select_images(root)
    if not image_files:
        messagebox.showerror("Erro", "Nenhuma imagem selecionada.")
        return

    show_images(image_files)

    dataset_dir = select_dataset_dir(root)
    if not dataset_dir:
        messagebox.showerror("Erro", "Nenhum diretório de dataset selecionado.")
        return

    class_name = simpledialog.askstring("Classe", "Digite o nome da classe para essas imagens:")
    if not class_name:
        messagebox.showerror("Erro", "Nome da classe não fornecido.")
        return

    dataset_type = simpledialog.askstring("Conjunto de Dados", "Digite o conjunto para essas imagens (train, validation, ou test):")
    if dataset_type not in ['train', 'validation', 'test']:
        messagebox.showerror("Erro", "Conjunto inválido. Use 'train', 'validation', ou 'test'.")
        return

    class_dir = os.path.join(dataset_dir, dataset_type, class_name)
    os.makedirs(class_dir, exist_ok=True)

    for image_path in image_files:
        filename = os.path.basename(image_path)
        destination = os.path.join(class_dir, filename)
        shutil.copy2(image_path, destination)

    messagebox.showinfo("Sucesso", f"{len(image_files)} imagens adicionadas à classe '{class_name}' no conjunto '{dataset_type}'.")

if __name__ == "__main__":
    organize_images_into_dataset()
