"""
=============================================================
  INTERPOLATION NUMÉRIQUE - Menu interactif
  Méthodes : Lagrange, Newton, Moindres Carrés,
             Polynomiale (numpy), Quelconque (spline)
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


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
    """Demande le degré du polynôme (pour moindres carrés / polynomiale)."""
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


def interpolation_lagrange(x_target, xi, yi):
    """Évalue le polynôme de Lagrange en x_target."""
    n = len(xi)
    result = 0.0
    for k in range(n):
        result += yi[k] * base_lagrange(x_target, xi, k)
    return result


def polynome_lagrange(x_vals, xi, yi):
    """Évalue le polynôme de Lagrange sur un tableau de valeurs."""
    return np.array([interpolation_lagrange(x, xi, yi) for x in x_vals])


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


def interpolation_newton(x_target, xi, table):
    """Évalue le polynôme de Newton en x_target."""
    n = len(xi)
    result = table[0, 0]
    produit = 1.0
    for k in range(1, n):
        produit *= (x_target - xi[k-1])
        result += table[0, k] * produit
    return result


def polynome_newton(x_vals, xi, table):
    """Évalue le polynôme de Newton sur un tableau de valeurs."""
    return np.array([interpolation_newton(x, xi, table) for x in x_vals])


# --------  MOINDRES CARRÉS --------

def moindres_carres(xi, yi, degre):
    """Ajustement polynomial au sens des moindres carrés."""
    coeffs = np.polyfit(xi, yi, degre)   # coefficients de degré haut vers bas
    poly   = np.poly1d(coeffs)
    return poly, coeffs


# --------  POLYNOMIALE (numpy polyfit exact si degré = n-1) --------

def interpolation_polynomiale(xi, yi, degre):
    """Polynôme de degré donné passant au mieux par les points."""
    coeffs = np.polyfit(xi, yi, degre)
    poly   = np.poly1d(coeffs)
    return poly, coeffs


# --------   MÉTHODE QUELCONQUE (Spline cubique) --------

def interpolation_spline(xi, yi):
    """Spline cubique naturelle — méthode quelconque."""
    spline = CubicSpline(xi, yi)
    return spline


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
#  SECTION 4 : MENU & CONTRÔLEUR PRINCIPAL
# ============================================================

MENU = """
         INTERPOLATION NUMÉRIQUE — Menu principal      

  1. Méthode de Lagrange                              
  2. Méthode de Newton (différences divisées)         
  3. Méthode des Moindres Carrés                      
  4. Méthode Polynomiale (numpy)                      
  5. Méthode Quelconque (Spline cubique)              
  0. Quitter                                          

"""


def run_lagrange():
    xi, yi = saisir_points()
    x_target = saisir_point_cible()
    valeur = interpolation_lagrange(x_target, xi, yi)
    afficher_resultat("Lagrange", x_target, valeur)

    # Courbe
    x_vals = np.linspace(xi.min(), xi.max(), 300)
    y_vals = polynome_lagrange(x_vals, xi, yi)
    afficher_courbe(xi, yi, x_vals, y_vals, "Lagrange")


def run_newton():
    xi, yi = saisir_points()
    x_target = saisir_point_cible()
    table = differences_divisees(xi, yi)
    valeur = interpolation_newton(x_target, xi, table)
    afficher_resultat("Newton", x_target, valeur)

    # Coefficients (diagonale supérieure de la table)
    coeffs_newton = [table[0, j] for j in range(len(xi))]
    print(f"\nCoefficients Newton (a0, a1, ...) : {np.round(coeffs_newton, 6)}")

    # Courbe
    x_vals = np.linspace(xi.min(), xi.max(), 300)
    y_vals = polynome_newton(x_vals, xi, table)
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


def run_polynomiale():
    xi, yi = saisir_points()
    degre = saisir_degre(len(xi) - 1)
    x_target = saisir_point_cible()
    poly, coeffs = interpolation_polynomiale(xi, yi, degre)
    valeur = poly(x_target)
    afficher_resultat("Polynomiale", x_target, valeur)
    afficher_coefficients(coeffs, "Coefficients (Polynomiale)")

    # Courbe
    x_vals = np.linspace(xi.min(), xi.max(), 300)
    afficher_courbe(xi, yi, x_vals, poly(x_vals), f"Polynomiale deg={degre}")


def run_spline():
    xi, yi = saisir_points()
    # Spline exige des xi triés
    ordre = np.argsort(xi)
    xi, yi = xi[ordre], yi[ordre]
    x_target = saisir_point_cible()
    spline = interpolation_spline(xi, yi)
    valeur = float(spline(x_target))
    afficher_resultat("Spline Cubique", x_target, valeur)

    # Courbe
    x_vals = np.linspace(xi.min(), xi.max(), 300)
    afficher_courbe(xi, yi, x_vals, spline(x_vals), "Spline Cubique")


ACTIONS = {
    '1': run_lagrange,
    '2': run_newton,
    '3': run_moindres_carres,
    '4': run_polynomiale,
    '5': run_spline,
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
