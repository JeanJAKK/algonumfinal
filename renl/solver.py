import numpy as np
import sympy as sp

# ============================================================
# VARIABLES GLOBALES
# ============================================================
x = sp.Symbol('x')  # La variable symbolique


# ============================================================
# FONCTIONS DE SAISIE
# ============================================================
def fonction():
    """Lecture et préparation de f(x)"""
    f_str = input("\nExpression de la fonction (ex: x**2 - 1): ")
    try:
        f_sympy = sp.sympify(f_str)
    except Exception as e:
        print("⚠ Erreur : fonction invalide.")
        print(f"  {e}")
        raise
    return f_sympy


def donnee():
    """Récupération des bornes"""
    while True:
        try:
            inf_str = input("Borne inférieure: ")
            supr_str = input("Borne supérieure: ")
            inf = float(inf_str)
            supr = float(supr_str)

            if supr <= inf:
                print("⚠ La borne supérieure doit être > à la borne inférieure.")
                continue

            return inf, supr
        except ValueError:
            print("⚠ Saisie invalide (valeur non numérique).")


def pas():
    """Récupération du pas de balayage"""
    while True:
        try:
            h_str = input("Pas de balayage h: ")
            h = float(h_str)
            if h <= 0:
                print("⚠ Le pas doit être > 0.")
                continue
            return h
        except ValueError:
            print("⚠ Saisie invalide.")


def precision():
    """Récupération de la précision"""
    while True:
        try:
            eps_str = input("Précision (exemple: 1e-7 ou 0.000001): ")
            return float(eps_str)
        except ValueError:
            print("⚠ La valeur n'est pas valide.")


def initial(methode="corde"):
    """Valeur initiale pour Newton-Raphson"""
    while True:
        try:
            xa_str = input(f"Initialisation x0 pour {methode}: ")
            xa = float(xa_str)
            return xa
        except ValueError:
            print("⚠ Saisie invalide.")


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
def balayage(f, inf, supr, h):
    """Recherche d'un intervalle de changement de signe [a, b]"""
    x0 = inf

    while x0 + h <= supr:
        try:
            y1 = f(x0)
            y2 = f(x0 + h)

            # Ignore NaN, infinies
            if np.isnan(y1) or np.isnan(y2) or np.isinf(y1) or np.isinf(y2):
                x0 += h
                continue

            if y1 * y2 < 0:   # changement de signe
                return x0, x0 + h

        except Exception:
            pass

        x0 += h

    return inf, supr


# ============================================================
# MÉTHODES DE RÉSOLUTION
# ============================================================
def ptfixe(f_sympy, a, b, eps):
    """Méthode du point fixe"""
    
    def generate_g_candidates(expr_str, lambda_val=0.1):
        g_candidates_expr = []
        g_candidates_expr.append(x - lambda_val * expr_str)
        try:
            sols = sp.solve(expr_str, x)
            for s in sols:
                g_candidates_expr.append(sp.simplify(s))
        except Exception:
            pass
        return g_candidates_expr

    def is_safe_g(g_expr, interval, num_points=200):
        try:
            g_num = sp.lambdify(x, g_expr, 'numpy')
            g_prime_expr = sp.diff(g_expr, x)
            g_prime_num = sp.lambdify(x, g_prime_expr, 'numpy')
            xs = np.linspace(interval[0], interval[1], num_points)
            for xi in xs:
                gi = g_num(xi)
                dpi = g_prime_num(xi)
                if np.isnan(gi) or np.isnan(dpi) or np.isinf(gi) or np.isinf(dpi):
                    return False
                if abs(dpi) >= 1:
                    return False
            return True
        except Exception:
            return False

    def filter_safe_g(g_candidates_expr, interval):
        return [g for g in g_candidates_expr if is_safe_g(g, interval)]

    def point_fixe(g_expr, x0, eps, max_iter=2000):
        g_num = sp.lambdify(x, g_expr, 'numpy')
        f_num = sp.lambdify(x, f_sympy, 'numpy')
        for _ in range(max_iter):
            try:
                x1 = g_num(x0)
                if np.isnan(x1) or np.isinf(x1):
                    return None
                if abs(x1 - x0) < eps:
                    return x1, _
                x0 = x1
            except Exception:
                return None
        return None

    print("\n--- Point Fixe ---")
    interval = (a, b)
    g_candidates = generate_g_candidates(f_sympy, lambda_val=0.1)
    safe_g = filter_safe_g(g_candidates, interval)

    if not safe_g:
        print("⚠ Aucune fonction g(x) valide (|g'(x)| < 1) trouvée.")
        return

    x0 = (interval[0] + interval[1]) / 2
    f_num = sp.lambdify(x, f_sympy, 'numpy')
    solution, nbre_itera = point_fixe(safe_g[0], x0, eps)

    if solution is None:
        print("⚠ Échec de convergence de la méthode du point fixe.")
        return

    print(f"✔ Racine approchée : x ≈ {solution}")
    print(f"   f({solution}) = {f_num(solution)}")
    print(f"Nombre d'itération : {nbre_itera}\n")


def dichosol(f_sympy, a, b, eps):
    """Méthode de la dichotomie"""
    f = sp.lambdify(x, f_sympy, "numpy")

    def dicho(a, b, eps):
        if f(a) * f(b) > 0:
            return None

        for _ in range(2000):
            if abs(b - a) <= eps:
                break

            m = (a + b) / 2

            try:
                fm = f(m)
            except Exception:
                return None

            if np.isnan(fm) or np.isinf(fm):
                return None

            if fm == 0:
                return m

            if f(a) * fm < 0:
                b = m
            else:
                a = m

        return (a + b) / 2, _

    print("\n--- Dichotomie ---")
    sol, nb_iter = dicho(a, b, eps)

    if sol is not None:
        print(f"✔ Racine approchée : x ≈ {sol}")
        print(f"   f({sol}) = {f(sol)}")
        print(f"Nombre d'itération : {nb_iter}\n")
    else:
        print("⚠ Aucune solution trouvée par dichotomie.")


def newsonsol(f_sympy, x0_init, eps):
    """Méthode de Newton-Raphson"""
    f_prime = sp.diff(f_sympy, x)
    f_prime_num = sp.lambdify(x, f_prime, 'numpy')
    f = sp.lambdify(x, f_sympy, "numpy")

    def newson(x0, eps, max_iter=1000):
        for _ in range(max_iter):
            try:
                fx0 = f(x0)
                fpx0 = f_prime_num(x0)

                if fpx0 == 0:
                    return None
                if np.isnan(fx0) or np.isinf(fx0) or np.isnan(fpx0) or np.isinf(fpx0):
                    return None

                x1 = x0 - fx0 / fpx0

                if abs(x1 - x0) < eps:
                    return x1, _
                x0 = x1
            except Exception:
                return None
        return x0, _

    print("\n--- Newton-Raphson ---")
    sol = newson(x0_init, eps)

    if sol is not None:
        print(f"✔ Racine approchée : x ≈ {sol}")
        print(f"   f({sol}) = {f(sol[0])}")
        print(f"Nombre d'itération : {sol[1]}\n")
    else:
        print("⚠ Échec de convergence de Newton-Raphson.")


def cordesol(f_sympy, a, b, eps):
    """Méthode de la corde"""
    f = sp.lambdify(x, f_sympy, "numpy")

    def corde(x0, x1, eps, max_iter=1000):
        if f(x0) * f(x1) > 0:
            return None

        for _ in range(max_iter):
            try:
                fx0 = f(x0)
                fx1 = f(x1)

                denominateur = fx1 - fx0
                if denominateur == 0:
                    return None

                x2 = x1 - fx1 * (x1 - x0) / denominateur

                if abs(x2 - x1) < eps:
                    return x2, _

                x0, x1 = x1, x2
            except Exception:
                return None

        return None

    print("\n--- Méthode de la Corde ---")
    sol = corde(a, b, eps)

    if sol is not None:
        print(f"✔ Racine approchée : x ≈ {sol}")
        print(f"   f({sol}) = {f(sol[0])}")
        print(f"Nombre d'itération : {sol[1]}\n")
    else:
        print("⚠ Échec de convergence ou pas de changement de signe.")


# ============================================================
# MENU PRINCIPAL
# ============================================================
def menu():
    """Menu principal pour la résolution d'équations non linéaires"""
    
    while True:
        print("\n" + "=" * 50)
        print("RÉSOLUTION D'ÉQUATION NON LINÉAIRE")
        print("=" * 50)
        
        print("\nMéthodes disponibles:")
        print("1 - Point Fixe")
        print("2 - Dichotomie")
        print("3 - Newton-Raphson")
        print("4 - Corde")
        print("5 - Toutes les méthodes")
        print("0 - Quitter")
        print("-" * 50)
        
        choix = input("\nVotre choix: ").strip()
        
        if choix == "0":
            print("\nAu revoir!")
            break
        
        # 1. Récupération des données
        try:
            f_sympy = fonction()
        except Exception:
            continue
            
        inf, supr = donnee()
        h = pas()
        
        # 2. Balayage
        f_num_balayage = sp.lambdify(x, f_sympy, 'numpy')
        resultat_balayage = balayage(f_num_balayage, inf, supr, h)
        
        if resultat_balayage is None:
            print("\n⚠ Aucun intervalle de changement de signe détecté. Modifiez les bornes ou le pas.")
            continue
        else:
            a, b = resultat_balayage
            print(f"\n✔ Intervalle détecté pour la racine : [{a}, {b}]")
        
        # 3. Récupération de la précision
        eps = precision()
        
        # 4. Pour Newton-Raphson
        x0_newton = None
        if choix == "3" or choix == "5":
            print(f"\n--- Initialisation de Newton (idéalement dans [{a}, {b}]) ---")
            x0_newton = initial(methode="Newton-Raphson")
        
        # 5. Pour Corde
        x0_newton = None
        if choix == "4" or choix == "5":
            print(f"\n--- Initialisation de Corde (idéalement dans [{a}, {b}]) ---")
            print(f"\n ---première initiale value ---" )
            x1_corde = initial()
            print(f"\n ---Deuxième initiale value ---" )
            x2_corde = initial()

        # 5. Appel des solveurs
        if choix == "1":
            ptfixe(f_sympy, a, b, eps)
        elif choix == "2":
            dichosol(f_sympy, a, b, eps)
        elif choix == "3":
            newsonsol(f_sympy, x0_newton, eps)
        elif choix == "4":
            cordesol(f_sympy, x1_corde, x2_corde , eps)
        elif choix == "5":
            ptfixe(f_sympy, a, b, eps)
            dichosol(f_sympy, a, b, eps)
            newsonsol(f_sympy, x0_newton, eps)
            cordesol(f_sympy, x1_corde, x2_corde , eps)
        else:
            print("\n⚠ Choix invalide!")


# ============================================================
# POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    menu()
