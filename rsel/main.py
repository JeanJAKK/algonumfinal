import numpy as np


def afficher_matrice(mat, titre="Matrice"):
    """Affiche une matrice de manière formatée"""
    print(f"\n{titre}:")
    for ligne in mat:
        print("  ", " ".join(f"{x:8.3f}" for x in ligne))
    print()


def gauss(A, b):
    """
    Méthode d'élimination de Gauss
    Résout le système Ax = b
    """
    n = len(b)
    # Créer la matrice augmentée
    M = np.column_stack([A.astype(float), b.astype(float)])

    print("=== MÉTHODE DE GAUSS ===")
    afficher_matrice(M, "Matrice augmentée initiale")

    # Élimination progressive (triangulation)
    for k in range(n - 1):
        for i in range(k + 1, n):
            if M[k, k] == 0:
                print("Erreur: pivot nul détecté!")
                return None

            facteur = M[i, k] / M[k, k]
            M[i, k:] = M[i, k:] - facteur * M[k, k:]

        afficher_matrice(M, f"Après élimination colonne {k + 1}")

    # Substitution arrière
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:n])) / M[i, i]

    return x


def gauss_jordan(A, b):
    """
    Méthode de Gauss-Jordan
    Résout le système Ax = b en réduisant à la forme identité
    """
    n = len(b)
    # Créer la matrice augmentée
    M = np.column_stack([A.astype(float), b.astype(float)])

    print("=== MÉTHODE DE GAUSS-JORDAN ===")
    afficher_matrice(M, "Matrice augmentée initiale")

    # Élimination de Gauss-Jordan
    for k in range(n):
        # Normalisation du pivot
        if M[k, k] == 0:
            print("Erreur: pivot nul détecté!")
            return None

        M[k, :] = M[k, :] / M[k, k]

        # Élimination sur toutes les lignes (sauf la ligne k)
        for i in range(n):
            if i != k:
                facteur = M[i, k]
                M[i, :] = M[i, :] - facteur * M[k, :]

        afficher_matrice(M, f"Après pivot ligne {k + 1}")

    # La solution est dans la dernière colonne
    x = M[:, -1]
    return x


def crout(A, b):
    """
    Décomposition de Crout (LU sans diagonale unitaire pour U)
    Résout le système Ax = b via LU = A
    """
    n = len(b)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    print("=== MÉTHODE DE CROUT ===")

    # Décomposition de Crout
    for j in range(n):
        # Diagonale de U = 1
        U[j, j] = 1

        # Calcul de la colonne j de L
        for i in range(j, n):
            somme = sum(L[i, k] * U[k, j] for k in range(j))
            L[i, j] = A[i, j] - somme

        # Calcul de la ligne j de U
        for i in range(j + 1, n):
            if L[j, j] == 0:
                print("Erreur: pivot nul dans la décomposition!")
                return None
            somme = sum(L[j, k] * U[k, i] for k in range(j))
            U[j, i] = (A[j, i] - somme) / L[j, j]

    afficher_matrice(L, "Matrice L")
    afficher_matrice(U, "Matrice U")

    # Résolution de Ly = b (substitution progressive)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    print("Vecteur y (solution de Ly = b):", y)

    # Résolution de Ux = y (substitution arrière)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = y[i] - np.dot(U[i, i + 1:], x[i + 1:])

    return x


def menu():
    """Menu principal"""
    print("\n" + "=" * 50)
    print("MÉTHODES DE TRANSFORMATION LINÉAIRE")
    print("=" * 50)

    # Exemple de système d'équations
    print("\nExemple de système : 3x + 2y - z = 1")
    print("                      2x - 2y + 4z = -2")
    print("                      -x + 0.5y - z = 0")

    A = np.array([[3, 2, -1],
                  [2, -2, 4],
                  [-1, 0.5, -1]], dtype=float)

    b = np.array([1, -2, 0], dtype=float)

    while True:
        print("\n" + "-" * 50)
        print("Choisissez une méthode:")
        print("1. Méthode de Gauss")
        print("2. Méthode de Gauss-Jordan")
        print("3. Méthode de Crout")
        print("4. Comparer toutes les méthodes")
        print("5. Entrer un nouveau système")
        print("0. Quitter")
        print("-" * 50)

        choix = input("\nVotre choix: ").strip()

        if choix == "0":
            print("\nAu revoir!")
            break

        elif choix == "1":
            x = gauss(A.copy(), b.copy())
            if x is not None:
                print("Solution:", x)
                print("Vérification Ax =", np.dot(A, x))

        elif choix == "2":
            x = gauss_jordan(A.copy(), b.copy())
            if x is not None:
                print("Solution:", x)
                print("Vérification Ax =", np.dot(A, x))

        elif choix == "3":
            x = crout(A.copy(), b.copy())
            if x is not None:
                print("Solution:", x)
                print("Vérification Ax =", np.dot(A, x))

        elif choix == "4":
            print("\n" + "=" * 50)
            print("COMPARAISON DES TROIS MÉTHODES")
            print("=" * 50)

            x1 = gauss(A.copy(), b.copy())
            x2 = gauss_jordan(A.copy(), b.copy())
            x3 = crout(A.copy(), b.copy())

            if all(x is not None for x in [x1, x2, x3]):
                print("\n" + "=" * 50)
                print("RÉSUMÉ DES SOLUTIONS")
                print("=" * 50)
                print(f"Gauss:        {x1}")
                print(f"Gauss-Jordan: {x2}")
                print(f"Crout:        {x3}")

        elif choix == "5":
            try:
                n = int(input("\nTaille du système (n): "))
                A = np.zeros((n, n))
                b = np.zeros(n)

                print("\nEntrez les coefficients de la matrice A:")
                for i in range(n):
                    for j in range(n):
                        A[i, j] = float(input(f"  A[{i + 1},{j + 1}] = "))

                print("\nEntrez le vecteur b:")
                for i in range(n):
                    b[i] = float(input(f"  b[{i + 1}] = "))

                print("\nNouveau système enregistré!")
                afficher_matrice(A, "Matrice A")
                print("Vecteur b:", b)

            except ValueError:
                print("\nErreur: Entrée invalide!")

        else:
            print("\nChoix invalide!")


if __name__ == "__main__":
    menu()