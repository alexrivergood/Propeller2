rom PIL import Image
import numpy as np
import os

def imagen_a_matriz(nombre_imagen="numero.png", normalizar=True, tamano=(28, 28)):

    ruta_imagen = os.path.join(os.path.dirname(__file__), nombre_imagen)

    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    img = Image.open(ruta_imagen).convert("L")   
    img = img.resize(tamano, resample=Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)

    
    arr = 255.0 - arr

    if normalizar:
        arr /= 255.0

    return arr   


def exportar_matriz_c(matriz, nombre_variable="numero"):
  
    lineas = []
    lineas.append(f"float {nombre_variable}[28][28] = {{")

    for fila in matriz:
        valores = ", ".join(f"{v:.4f}" for v in fila)
        lineas.append(f"    {{ {valores} }},")
    lineas.append("};\n")

    return "\n".join(lineas)


def main():
    nombre_imagen = "numero.png"
    matriz = imagen_a_matriz(nombre_imagen)

    print("Imagen convertida a matriz 28x28.")

    c_code = exportar_matriz_c(matriz)

    salida = os.path.join(os.path.dirname(__file__), "numero_matriz.h")
    with open(salida, "w") as f:
        f.write(c_code)

    print(f"Archivo generado: {salida}")
    print("\nPrimeras filas:\n")
    print("\n".join(c_code.splitlines()[:8]))


if __name__ == "__main__":
    main()
