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
#  SECTION 1 : SAISIE DES DONNÉES
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
#  SECTION 2 : TRAITEMENTS — LES MÉTHODES
# ============================================================

# -------- 2.1  LAGRANGE --------

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


# -------- 2.2  NEWTON (différences divisées) --------

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


# -------- 2.3  MOINDRES CARRÉS --------

def moindres_carres(xi, yi, degre):
    """Ajustement polynomial au sens des moindres carrés."""
    coeffs = np.polyfit(xi, yi, degre)   # coefficients de degré haut vers bas
    poly   = np.poly1d(coeffs)
    return poly, coeffs


# -------- 2.4  POLYNOMIALE (numpy polyfit exact si degré = n-1) --------

def interpolation_polynomiale(xi, yi, degre):
    """Polynôme de degré donné passant au mieux par les points."""
    coeffs = np.polyfit(xi, yi, degre)
    poly   = np.poly1d(coeffs)
    return poly, coeffs


# -------- 2.5  MÉTHODE QUELCONQUE (Spline cubique) --------

def interpolation_spline(xi, yi):
    """Spline cubique naturelle — méthode quelconque."""
    spline = CubicSpline(xi, yi)
    return spline


# ============================================================
#  SECTION 3 : EXPRESSION SYMBOLIQUE DU POLYNÔME TROUVÉ
# ============================================================

def expression_depuis_coeffs(coeffs, label="P"):
    """
    Construit et retourne la chaîne P(x) = a_n*x^n + ... + a_1*x + a_0
    à partir de coefficients numpy (ordre décroissant).
    """
    degre = len(coeffs) - 1
    termes = []
    for i, c in enumerate(coeffs):
        puissance = degre - i
        c_arrondi = round(float(c), 6)
        if abs(c_arrondi) < 1e-10:
            continue  # terme nul, on l'ignore
        signe = "+ " if c_arrondi >= 0 else "- "
        valeur_abs = abs(c_arrondi)
        if puissance == 0:
            termes.append(f"{signe}{valeur_abs}")
        elif puissance == 1:
            termes.append(f"{signe}{valeur_abs}·x")
        else:
            termes.append(f"{signe}{valeur_abs}·x^{puissance}")
    if not termes:
        return f"{label}(x) = 0"
    # Retirer le "+" initial si le premier terme est positif
    expr = " ".join(termes).lstrip("+ ")
    return f"{label}(x) = {expr}"


def expression_lagrange(xi, yi):
    """
    Convertit le polynôme de Lagrange en coefficients standard
    via numpy, puis retourne son expression.
    """
    n = len(xi)
    # On évalue sur n+1 points équidistants et on refit à degré n-1
    x_eval = np.linspace(xi.min(), xi.max(), max(50, n * 10))
    y_eval = polynome_lagrange(x_eval, xi, yi)
    coeffs = np.polyfit(x_eval, y_eval, n - 1)
    return expression_depuis_coeffs(coeffs, "P_Lagrange")


def expression_newton(xi, table):
    """
    Convertit le polynôme de Newton en coefficients standard
    via numpy, puis retourne son expression.
    """
    n = len(xi)
    x_eval = np.linspace(xi.min(), xi.max(), max(50, n * 10))
    y_eval = polynome_newton(x_eval, xi, table)
    coeffs = np.polyfit(x_eval, y_eval, n - 1)
    return expression_depuis_coeffs(coeffs, "P_Newton")


def expression_spline(xi, spline):
    """
    Affiche les équations cubiques morceau par morceau de la spline.
    Retourne une liste de chaînes, une par intervalle.
    """
    lignes = []
    for i in range(len(xi) - 1):
        # scipy stocke les coefficients par rapport à (x - xi[i])
        c = spline.c[:, i]   # [c3, c2, c1, c0] ordre décroissant
        a3, a2, a1, a0 = round(float(c[0]), 6), round(float(c[1]), 6), \
                          round(float(c[2]), 6), round(float(c[3]), 6)
        expr = (f"S_{i}(x) = {a3}·(x-{xi[i]})^3 + {a2}·(x-{xi[i]})^2 "
                f"+ {a1}·(x-{xi[i]}) + {a0}  "
                f"  sur [{xi[i]}, {xi[i+1]}]")
        lignes.append(expr)
    return lignes


# ============================================================
#  SECTION 4 : AFFICHAGE DES RÉSULTATS
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
╔══════════════════════════════════════════════════════╗
║        INTERPOLATION NUMÉRIQUE — Menu principal      ║
╠══════════════════════════════════════════════════════╣
║  1. Méthode de Lagrange                              ║
║  2. Méthode de Newton (différences divisées)         ║
║  3. Méthode des Moindres Carrés                      ║
║  4. Méthode Polynomiale (numpy)                      ║
║  5. Méthode Quelconque (Spline cubique)              ║
║  0. Quitter                                          ║
╚══════════════════════════════════════════════════════╝
"""


def run_lagrange():
    xi, yi = saisir_points()
    x_target = saisir_point_cible()
    valeur = interpolation_lagrange(x_target, xi, yi)
    afficher_resultat("Lagrange", x_target, valeur)

    # Expression symbolique du polynôme
    print("\n--- Fonction trouvée ---")
    print(expression_lagrange(xi, yi))

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

    # Expression symbolique du polynôme
    print("\n--- Fonction trouvée ---")
    print(expression_newton(xi, table))

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

    # Expression symbolique
    print("\n--- Fonction trouvée ---")
    print(expression_depuis_coeffs(coeffs, "P_MC"))

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

    # Expression symbolique
    print("\n--- Fonction trouvée ---")
    print(expression_depuis_coeffs(coeffs, "P_poly"))

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

    # Expression morceau par morceau
    print("\n--- Fonction trouvée (morceaux cubiques) ---")
    for ligne in expression_spline(xi, spline):
        print(" ", ligne)

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