"""
============================================================
  INTERPOLATION NUMÉRIQUE - Menu interactif
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SAISIE DES DONNÉES
# ============================================================

def saisir_points():
    """Demande à l'utilisateur de saisir les points (xi, yi)."""
    print("\n--- Saisie des points d'interpolation ---")
    n = int(input("Nombre de points : "))
    x = []
    y = []
    for i in range(n):
        xi = float(input(f"  x[{i}] = "))
        yi = float(input(f"  y[{i}] = "))
        x.append(xi)
        y.append(yi)
    return np.array(x), np.array(y)


def saisir_point_cible():
    """Demande la valeur de x en laquelle évaluer l'interpolation."""
    return float(input("Valeur de x à estimer : "))


def saisir_degre(max_degre):
    """Demande le degré du polynôme (pour moindres carrés)."""
    d = int(input(f"Degré du polynôme (1 à {max_degre}) : "))
    return max(1, min(d, max_degre))


# ============================================================
# TRAITEMENTS — LES MÉTHODES
# ============================================================

# --------   LAGRANGE --------

def base_lagrange(x, xi, k):
    """Calcule la k-ième base de Lagrange L_k(x)."""
    n = len(xi)
    L = 1.0
    for j in range(n):
        if j != k:
            L *= (x - xi[j]) / (xi[k] - xi[j])
    return L


def interpolation_lagrange(xi, yi):
    """Retourne la fonction polynomiale de Lagrange interpolant les points.

    La fonction renvoyée accepte des scalaires ou des tableaux numpy et
    effectue le calcul en utilisant la base de Lagrange.
    """
    def poly(x):
        n = len(xi)
        result = 0.0
        for k in range(n):
            result += yi[k] * base_lagrange(x, xi, k)
        return result
    return poly


def polynome_lagrange(x_vals, xi, yi):
    """Évalue le polynôme de Lagrange sur un tableau de valeurs.

    Ne garde la fonction historique que pour la compatibilité interne.
    """
    poly = interpolation_lagrange(xi, yi)
    return poly(x_vals)


def polynome_lagrange_str(xi, yi):
    """Retourne la représentation polynomiale de Lagrange sous forme de chaîne.
    
    Développe le polynôme d'interpolation de Lagrange en forme standard
    P(x) = a_n*x^n + ... + a_1*x + a_0
    """
    n = len(xi)
    # Créer un vecteur de coefficients en initialisant à 0
    coeffs = np.zeros(n)
    
    # Pour chaque base de Lagrange L_k(x)
    for k in range(n):
        # Calculer les coefficients du polynôme L_k(x)
        # L_k(x) = produit de (x - x_j)/(x_k - x_j) pour j != k
        
        # Commencer avec le polynôme [1]
        L_k_coeffs = np.array([1.0])
        
        # Multiplier par (x - x_j) / (x_k - x_j) pour chaque j != k
        for j in range(n):
            if j != k:
                # Diviser par (x_k - x_j)
                scale = 1.0 / (xi[k] - xi[j])
                # Créer le polynôme (x - x_j)
                poly_x_minus_xj = np.array([scale, -scale * xi[j]])
                # Multiplier les polynômes
                L_k_coeffs = np.polymul(L_k_coeffs, poly_x_minus_xj)
        
        # Ajouter y_k * L_k au polynôme total
        # Ajuster la taille des vecteurs si nécessaire
        if len(L_k_coeffs) > len(coeffs):
            coeffs = np.pad(coeffs, (len(L_k_coeffs) - len(coeffs), 0))
        coeffs[-(len(L_k_coeffs)):] += yi[k] * L_k_coeffs
    
    # Créer une représentation lisible
    degree = len(coeffs) - 1
    terms = []
    
    for i, coeff in enumerate(coeffs):
        power = degree - i
        # Ignorer les termes négligeables
        if abs(coeff) < 1e-10:
            continue
        
        # Formater le coefficient
        if power == 0:
            terms.append(f"{coeff:.6f}")
        elif power == 1:
            if abs(coeff - 1.0) < 1e-10:
                terms.append("x")
            elif abs(coeff + 1.0) < 1e-10:
                terms.append("-x")
            else:
                terms.append(f"{coeff:.6f}*x")
        else:
            if abs(coeff - 1.0) < 1e-10:
                terms.append(f"x^{power}")
            elif abs(coeff + 1.0) < 1e-10:
                terms.append(f"-x^{power}")
            else:
                terms.append(f"{coeff:.6f}*x^{power}")
    
    # Assembler la chaîne
    if not terms:
        return "0"
    
    result = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            result += " " + term
        else:
            result += " + " + term
    
    return result


# --------  NEWTON (différences divisées) --------

def differences_divisees(xi, yi):
    """Calcule la table des différences divisées."""
    n = len(xi)
    table = np.zeros((n, n))
    table[:, 0] = yi
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i+1, j-1] - table[i, j-1]) / (xi[i+j] - xi[i])
    return table


def interpolation_newton(xi, yi):
    """Construit le polynôme de Newton interpolant et renvoie (poly, table).

    * ``poly`` est une fonction qui accepte des scalaires ou tableaux.
    * ``table`` est la table des différences divisées, utile pour
      extraire les coefficients.
    """
    table = differences_divisees(xi, yi)
    def poly(x):
        n = len(xi)
        result = table[0, 0]
        produit = 1.0
        for k in range(1, n):
            produit *= (x - xi[k-1])
            result += table[0, k] * produit
        return result
    return poly, table


def polynome_newton(x_vals, xi, table):
    """Évalue le polynôme de Newton sur un tableau de valeurs.

    Recalcule la valeur en utilisant la table des différences divisées.
    """
    def poly(x):
        n = len(xi)
        result = table[0, 0]
        produit = 1.0
        for k in range(1, n):
            produit *= (x - xi[k-1])
            result += table[0, k] * produit
        return result
    return poly(x_vals)


# --------  MOINDRES CARRÉS --------

def moindres_carres(xi, yi, degre):
    """Ajustement polynomial au sens des moindres carrés."""
    coeffs = np.polyfit(xi, yi, degre)   # coefficients de degré haut vers bas
    poly   = np.poly1d(coeffs)
    return poly, coeffs



# ============================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================

def afficher_resultat(methode, x_target, valeur):
    print(f"\n[{methode}]  P({x_target}) ≈ {valeur:.6f}")


def afficher_courbe(xi, yi, x_vals, y_vals, methode):
    """Trace la courbe d'interpolation avec les points de données."""
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, 'b-', label=f'Interpolation {methode}')
    plt.scatter(xi, yi, color='red', zorder=5, label='Points donnés')
    plt.title(f'Interpolation — {methode}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def afficher_coefficients(coeffs, label="Coefficients"):
    print(f"\n{label} (degré haut → bas) : {np.round(coeffs, 6)}")


# ============================================================
# MENU & CONTRÔLEUR PRINCIPAL
# ============================================================

MENU = """
         INTERPOLATION NUMÉRIQUE — Menu principal      

  1. Méthode de Lagrange                              
  2. Méthode de Newton (différences divisées)         
  3. Méthode des Moindres Carrés                      
  0. Quitter                                          

"""

def run_lagrange():
    xi, yi = saisir_points()
    x_target = saisir_point_cible()
    # construire la fonction d'interpolation
    poly = interpolation_lagrange(xi, yi)
    valeur = poly(x_target)
    afficher_resultat("Lagrange", x_target, valeur)
    
    # Afficher le polynôme en forme développée
    poly_str = polynome_lagrange_str(xi, yi)
    print(f"\nPolynôme : P(x) = {poly_str}")

    # Courbe
    x_vals = np.linspace(xi.min(), xi.max(), 300)
    y_vals = poly(x_vals)
    afficher_courbe(xi, yi, x_vals, y_vals, "Lagrange")


def run_newton():
    xi, yi = saisir_points()
    x_target = saisir_point_cible()
    poly, table = interpolation_newton(xi, yi)
    valeur = poly(x_target)
    afficher_resultat("Newton", x_target, valeur)

    # Coefficients (diagonale supérieure de la table)
    coeffs_newton = [table[0, j] for j in range(len(xi))]
    print(f"\nCoefficients Newton (a0, a1, ...) : {np.round(coeffs_newton, 6)}")

    # Courbe
    x_vals = np.linspace(xi.min(), xi.max(), 300)
    y_vals = poly(x_vals)
    afficher_courbe(xi, yi, x_vals, y_vals, "Newton")


def run_moindres_carres():
    xi, yi = saisir_points()
    degre = saisir_degre(len(xi) - 1)
    x_target = saisir_point_cible()
    poly, coeffs = moindres_carres(xi, yi, degre)
    valeur = poly(x_target)
    afficher_resultat("Moindres Carrés", x_target, valeur)
    afficher_coefficients(coeffs, "Coefficients (Moindres Carrés)")

    # Erreur quadratique
    residus = yi - poly(xi)
    print(f"Erreur quadratique totale : {np.sum(residus**2):.6f}")

    # Courbe
    x_vals = np.linspace(xi.min(), xi.max(), 300)
    afficher_courbe(xi, yi, x_vals, poly(x_vals), f"Moindres Carrés deg={degre}")



ACTIONS = {
    '1': run_lagrange,
    '2': run_newton,
    '3': run_moindres_carres,
}


def main():
    print("\n  Bienvenue dans le programme d'Interpolation Numérique")
    while True:
        print(MENU)
        choix = input("Votre choix : ").strip()
        if choix == '0':
            print("\n  Au revoir !\n")
            break
        action = ACTIONS.get(choix)
        if action:
            try:
                action()
            except Exception as e:
                print(f"\n  Erreur : {e}")
        else:
            print("  Choix invalide, veuillez réessayer.")


if __name__ == "__main__":
    main()
