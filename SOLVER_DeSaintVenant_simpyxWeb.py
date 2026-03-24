import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Arc
import sympy as sp
import io
from contextlib import redirect_stdout

def sym_ex(expr):
    if expr is None or str(expr).strip() == "":
        return None
    if isinstance(expr, str):
        # Diciamo a SymPy quali variabili sono strettamente POSITIVE 
        # (fondamentale per semplificare le radici quadrate delle lunghezze!)
        variabili_sicure = {
            'N': sp.Symbol('N'), 
            'T': sp.Symbol('T'),
            'I': sp.Symbol('I'), 
            'E': sp.Symbol('E'),
            'b': sp.Symbol('b', positive=True), # <--- ECCO IL TRUCCO
            't': sp.Symbol('t', positive=True)  # <--- ECCO IL TRUCCO
        }
        expr = sp.sympify(expr, locals=variabili_sicure)
    
    return sp.nsimplify(sp.simplify(expr), rational=True)

class Node:
    def __init__(self, node_id, x, y):
        self.id = node_id
        self.x = sym_ex(x)
        self.y = sym_ex(y)
        self.connected_elements = []

    def __repr__(self):
        return f"Node({self.id}, x={self.x}, y={self.y})"

class Element:
    def __init__(self, elem_id, node_start, node_end, thickness):
        self.id = elem_id
        self.n1 = node_start
        self.n2 = node_end
        self.t = sym_ex(thickness)
        
        self.n1.connected_elements.append(self)
        self.n2.connected_elements.append(self)

    @property
    def length(self):
        return sym_ex(sp.sqrt((self.n2.x - self.n1.x)**2 + (self.n2.y - self.n1.y)**2))

    @property
    def area(self):
        return sym_ex(self.length * self.t)

    @property
    def center_of_gravity(self):
        # Uso sp.Rational(1,2) al posto di / 2 per evitare la nascita di float
        return (sym_ex((self.n1.x + self.n2.x) * sp.Rational(1, 2)), 
                sym_ex((self.n1.y + self.n2.y) * sp.Rational(1, 2)))

class Load:
    def __init__(self, load_type, value, x_app=None, y_app=None):
        self.type = load_type.upper() 
        self.value = sym_ex(value)
        self.x = sym_ex(x_app) if x_app is not None else None
        self.y = sym_ex(y_app) if y_app is not None else None

class CrossSection:
    def __init__(self, nodes, elements):
        self.nodes = {n.id: n for n in nodes}
        self.elements = elements
        
        self.A = 0
        self.x_G, self.y_G = 0, 0
        self.I_x, self.I_y, self.I_xy = 0, 0, 0
        self.x_CT, self.y_CT = 0, 0
        
        self.compute_geometric_properties()
        self.compute_shear_center()

    def ev(self, expr, subs_dict):
        """Helper che converte temporaneamente l'espressione in float numerico SOLO per i grafici Matplotlib"""
        if expr is None: return 0.0
        try:
            return float(sp.sympify(expr).evalf(subs=subs_dict))
        except:
            return 0.0

    def compute_geometric_properties(self):
        Sx, Sy = 0, 0
        for el in self.elements:
            self.A += el.area
            cx, cy = el.center_of_gravity
            Sx += sym_ex(el.area * cy)
            Sy += sym_ex(el.area * cx)
            
        self.A = sym_ex(self.A)
        self.x_G = sym_ex(Sy / self.A)
        self.y_G = sym_ex(Sx / self.A)

        for el in self.elements:
            dx = sym_ex(el.n2.x - el.n1.x)
            dy = sym_ex(el.n2.y - el.n1.y)
            L = el.length
            A_el = el.area
            
            I_cx = sym_ex(sp.Rational(1,12) * A_el * dy**2)
            I_cy = sym_ex(sp.Rational(1,12) * A_el * dx**2)
            I_cxy = sym_ex(sp.Rational(1,12) * A_el * dx * dy)
            
            cx, cy = el.center_of_gravity
            dist_x = sym_ex(cx - self.x_G)
            dist_y = sym_ex(cy - self.y_G)
            
            self.I_x += sym_ex(I_cx + A_el * dist_y**2)
            self.I_y += sym_ex(I_cy + A_el * dist_x**2)
            self.I_xy += sym_ex(I_cxy + A_el * dist_x * dist_y)
            
        self.I_x = sym_ex(self.I_x)
        self.I_y = sym_ex(self.I_y)
        self.I_xy = sym_ex(self.I_xy)

    def compute_shear_center(self):
        Delta = sym_ex(self.I_x * self.I_y - self.I_xy**2)
        if Delta == 0:
            self.x_CT, self.y_CT = self.x_G, self.y_G
            return 
            
        M_G_from_Ty = 0 
        M_G_from_Tx = 0 
        
        degree = {n_id: len(n.connected_elements) for n_id, n in self.nodes.items()}
        leaves = [n_id for n_id, deg in degree.items() if deg == 1]
        
        node_Sx = {n_id: 0 for n_id in self.nodes}
        node_Sy = {n_id: 0 for n_id in self.nodes}
        
        processed_elements = set()
        
        while leaves:
            curr_node_id = leaves.pop(0)
            curr_node = self.nodes[curr_node_id]
            
            unprocessed_elems = [e for e in curr_node.connected_elements if e.id not in processed_elements]
            if not unprocessed_elems: continue
                
            el = unprocessed_elems[0]
            processed_elements.add(el.id)
            next_node = el.n1 if el.n2 == curr_node else el.n2
            
            x1 = sym_ex(curr_node.x - self.x_G)
            y1 = sym_ex(curr_node.y - self.y_G)
            x2 = sym_ex(next_node.x - self.x_G)
            y2 = sym_ex(next_node.y - self.y_G)
            
            S_x_start = node_Sx[curr_node_id]
            S_y_start = node_Sy[curr_node_id]
            
            S_x_mean = sym_ex(S_x_start + el.area * sp.Rational(1, 6) * (2*y1 + y2))
            S_y_mean = sym_ex(S_y_start + el.area * sp.Rational(1, 6) * (2*x1 + x2))
            
            q_mean_from_Tx = sym_ex((self.I_xy * S_x_mean - self.I_x * S_y_mean) / Delta)
            q_mean_from_Ty = sym_ex((self.I_xy * S_y_mean - self.I_y * S_x_mean) / Delta)
            
            r_n = sym_ex((x1 * y2 - x2 * y1) / el.length)
            
            M_G_from_Tx += sym_ex(q_mean_from_Tx * el.length * r_n)
            M_G_from_Ty += sym_ex(q_mean_from_Ty * el.length * r_n)
            
            node_Sx[next_node.id] += sym_ex(S_x_start + el.area * sp.Rational(1, 2) * (y1 + y2))
            node_Sy[next_node.id] += sym_ex(S_y_start + el.area * sp.Rational(1, 2) * (x1 + x2))
            
            degree[next_node.id] -= 1
            if degree[next_node.id] == 1:
                leaves.append(next_node.id)

        self.x_CT = sym_ex(self.x_G + M_G_from_Ty)
        self.y_CT = sym_ex(self.y_G - M_G_from_Tx)

    def print_recap(self):
        print("=== RECAP PROPRIETÀ GEOMETRICHE E INERZIALI (SIMBOLICHE) ===")
        print(f"Area Totale (A) : {self.A}")
        print(f"Baricentro (G)  : XG = {self.x_G}, YG = {self.y_G}")
        print(f"Inerzia Ix      : {self.I_x}")
        print(f"Inerzia Iy      : {self.I_y}")
        print(f"Inerzia Ixy     : {self.I_xy}")
        print(f"Centro Taglio   : X_CT = {self.x_CT}, Y_CT = {self.y_CT}")
        print("============================================================\n")

    def _find_fundamental_cycle(self):
        visited = set()
        path = []
        def dfs(curr_node, parent_node):
            visited.add(curr_node)
            path.append(curr_node)
            for el in curr_node.connected_elements:
                neighbor = el.n1 if el.n2 == curr_node else el.n2
                if neighbor == parent_node: continue
                if neighbor in visited:
                    cycle_start_idx = path.index(neighbor)
                    return path[cycle_start_idx:]
                res = dfs(neighbor, curr_node)
                if res: return res
            path.pop()
            return None
        for start_node in self.nodes.values():
            if start_node not in visited:
                cycle_nodes = dfs(start_node, None)
                if cycle_nodes: return cycle_nodes
        return None

    def _get_elements_in_cycle(self, cycle_nodes):
        cycle_elems = []
        for i in range(len(cycle_nodes)):
            n_curr = cycle_nodes[i]
            n_next = cycle_nodes[(i + 1) % len(cycle_nodes)]
            for el in n_curr.connected_elements:
                if (el.n1 == n_next or el.n2 == n_next):
                    if el not in cycle_elems: cycle_elems.append(el)
                    break
        return cycle_elems

    # ==========================================================
    # LOGICA GRAFICA 1: RIEPILOGO CARICHI
    # ==========================================================
    def plot_loads_summary(self, loads, ax, subs_dict):
        for el in self.elements:
            ax.plot([self.ev(el.n1.x, subs_dict), self.ev(el.n2.x, subs_dict)], 
                    [self.ev(el.n1.y, subs_dict), self.ev(el.n2.y, subs_dict)], 
                    'k-', linewidth=self.ev(el.t, subs_dict), solid_capstyle='round', alpha=0.3)
        
        xg, yg = self.ev(self.x_G, subs_dict), self.ev(self.y_G, subs_dict)
        xct, yct = self.ev(self.x_CT, subs_dict), self.ev(self.y_CT, subs_dict)
        
        ax.plot(xg, yg, 'ko', markersize=6)
        ax.text(xg + 2, yg - 2, 'G', fontsize=12, color='black')
        
        ax.plot(xct, yct, 'g*', markersize=10)
        ax.text(xct + 2, yct + 5, r'$C_T$', fontsize=12, color='green')

        x_vals = [self.ev(n.x, subs_dict) for n in self.nodes.values()]
        y_vals = [self.ev(n.y, subs_dict) for n in self.nodes.values()]
        w = max(x_vals) - min(x_vals) if x_vals else 100
        h = max(y_vals) - min(y_vals) if y_vals else 100
        arr_len = max(w, h) * 0.15

        for load in loads:
            x_app_sym = load.x if load.x is not None else self.x_G
            y_app_sym = load.y if load.y is not None else self.y_G
            x_app = self.ev(x_app_sym, subs_dict)
            y_app = self.ev(y_app_sym, subs_dict)
            val_num = self.ev(load.value, subs_dict)
            lbl = load.value
            
            if load.type == 'N':
                marker = 'o' if val_num > 0 else 'X'
                color = 'red' if val_num > 0 else 'blue'
                ax.plot(x_app, y_app, marker=marker, markersize=10, color=color, markeredgecolor='black')
                ax.text(x_app + arr_len*0.3, y_app, f"N={lbl}", color=color, fontsize=10)
                
            elif load.type == 'TX':
                dir_x = 1 if val_num > 0 else -1
                x_start = x_app - dir_x * arr_len if load.x is not None else xct - dir_x * arr_len
                y_pos = y_app if load.y is not None else yct
                ax.arrow(x_start, y_pos, dir_x * arr_len, 0, head_width=arr_len*0.3, color='orange', zorder=5)
                ax.text(x_start, y_pos - arr_len*0.3, f"Tx={lbl}", color='orange', fontsize=10)
                
            elif load.type == 'TY':
                dir_y = 1 if val_num > 0 else -1
                y_start = y_app - dir_y * arr_len if load.y is not None else yct - dir_y * arr_len
                x_pos = x_app if load.x is not None else xct
                ax.arrow(x_pos, y_start, 0, dir_y * arr_len, head_width=arr_len*0.3, color='purple', zorder=5)
                ax.text(x_pos + arr_len*0.2, y_start, f"Ty={lbl}", color='purple', fontsize=10)

            elif load.type == 'MT':
                arc = Arc((x_app, y_app), arr_len, arr_len, angle=0, theta1=0, theta2=270, color='brown', lw=2)
                ax.add_patch(arc)
                ax.text(x_app + arr_len*0.5, y_app + arr_len*0.5, f"Mt={lbl}", color='brown', fontsize=10)
                
            elif load.type in ['MX', 'MY']:
                ax.text(x_app, y_app + arr_len*0.5, f"{load.type}={lbl}", color='darkred', fontsize=10)

        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title("Schema Sezione e Carichi Applicati", fontsize=14)

    # ==========================================================
    # LOGICA GRAFICA 2: NAVIER
    # ==========================================================
    def solve_navier(self, loads, ax, subs_dict):
        N_tot, Mx_tot, My_tot = 0, 0, 0 
        
        for load in loads:
            if load.type == 'N':
                N_tot += load.value
                if load.x is not None and load.y is not None:
                    dist_x = sym_ex(load.x - self.x_G)
                    dist_y = sym_ex(load.y - self.y_G)
                    Mx_tot += sym_ex(load.value * dist_y)
                    My_tot += sym_ex(-load.value * dist_x)
            elif load.type == 'MX':
                Mx_tot += load.value
            elif load.type == 'MY':
                My_tot += load.value
                
        Delta = sym_ex(self.I_x * self.I_y - self.I_xy**2)
        c_y = sym_ex((Mx_tot * self.I_y + My_tot * self.I_xy) / Delta) if Delta != 0 else 0
        c_x = sym_ex((My_tot * self.I_x + Mx_tot * self.I_xy) / Delta) if Delta != 0 else 0
        
        def get_sigma(x, y):
            x_rel = sym_ex(x - self.x_G)
            y_rel = sym_ex(y - self.y_G)
            return sym_ex(N_tot / self.A + c_y * y_rel - c_x * x_rel)

        sigmas_num = {n_id: self.ev(get_sigma(n.x, n.y), subs_dict) for n_id, n in self.nodes.items()}
        if all(abs(v) < 1e-9 for v in sigmas_num.values()):
            ax.axis('off')
            ax.set_title("Navier: Nessun carico assiale/flettente", fontsize=12)
            return

        max_node_id = max(sigmas_num, key=sigmas_num.get)
        min_node_id = min(sigmas_num, key=sigmas_num.get)
        
        sigma_max_sym = get_sigma(self.nodes[max_node_id].x, self.nodes[max_node_id].y)
        sigma_min_sym = get_sigma(self.nodes[min_node_id].x, self.nodes[min_node_id].y)

        print("=== RISULTATI NAVIER ===")
        print(f"Sforzo Normale Totale (N) = {sym_ex(N_tot)}")
        print(f"Momento Flettente (Mx) al Baricentro = {sym_ex(Mx_tot)}")
        print(f"Momento Flettente (My) al Baricentro = {sym_ex(My_tot)}")
        print(f"Tensione Max (Trazione): {sigma_max_sym} (Nodo {max_node_id})")
        print(f"Tensione Min (Compressione): {sigma_min_sym} (Nodo {min_node_id})\n")

        for el in self.elements:
            ax.plot([self.ev(el.n1.x, subs_dict), self.ev(el.n2.x, subs_dict)], 
                    [self.ev(el.n1.y, subs_dict), self.ev(el.n2.y, subs_dict)], 
                    'k-', linewidth=self.ev(el.t, subs_dict)*1.5, solid_capstyle='round')

        xg = self.ev(self.x_G, subs_dict)
        yg = self.ev(self.y_G, subs_dict)
        x_vals = [self.ev(n.x, subs_dict) for n in self.nodes.values()]
        y_vals = [self.ev(n.y, subs_dict) for n in self.nodes.values()]
        w = max(x_vals) - min(x_vals)
        h = max(y_vals) - min(y_vals)

        ax.plot(xg, yg, 'ko') 
        ax.text(xg + 2, yg - 2, 'G', fontsize=12)
        
        c_y_num = self.ev(c_y, subs_dict)
        c_x_num = self.ev(c_x, subs_dict)
        N_num = self.ev(N_tot, subs_dict)
        A_num = self.ev(self.A, subs_dict)

        m_na = c_x_num / c_y_num if abs(c_y_num) > 1e-9 else np.inf
        y0_rel = -(N_num / A_num) / c_y_num if abs(c_y_num) > 1e-9 else 0
        x_line = np.array([min(x_vals) - w*0.5, max(x_vals) + w*0.5])
        
        if m_na != np.inf:
            y_line = m_na * (x_line - xg) + y0_rel + yg
            ax.plot(x_line, y_line, 'k:', linewidth=1.5)
            ax.text(x_line[0], y_line[0]-2, 'n', fontsize=14)
            ax.text(x_line[-1], y_line[-1]+2, 'n', fontsize=14)

            u_dir = np.array([1, m_na]) / np.sqrt(1 + m_na**2)
            v_dir = np.array([-m_na, 1]) / np.sqrt(1 + m_na**2)
        else:
            u_dir, v_dir = np.array([0, 1]), np.array([-1, 0])

        if u_dir[1] > 0: u_dir = -u_dir

        n_max, n_min = self.nodes[max_node_id], self.nodes[min_node_id]
        get_v = lambda nx, ny: np.dot([nx - xg, ny - yg], v_dir)
        
        u_vals = [np.dot([self.ev(n.x, subs_dict) - xg, self.ev(n.y, subs_dict) - yg], u_dir) for n in self.nodes.values()]
        u_offset = max(u_vals) + h * 0.5 

        B_start = np.array([xg, yg]) + u_offset * u_dir + get_v(self.ev(n_max.x, subs_dict), self.ev(n_max.y, subs_dict)) * v_dir
        B_end = np.array([xg, yg]) + u_offset * u_dir + get_v(self.ev(n_min.x, subs_dict), self.ev(n_min.y, subs_dict)) * v_dir

        sig_max_num, sig_min_num = sigmas_num[max_node_id], sigmas_num[min_node_id]
        scale = (w * 0.75) / max(abs(sig_max_num), abs(sig_min_num)) if max(abs(sig_max_num), abs(sig_min_num)) != 0 else 1

        V_start = B_start + (sig_max_num * scale) * u_dir
        V_end = B_end + (sig_min_num * scale) * u_dir

        ax.plot([B_start[0], B_end[0]], [B_start[1], B_end[1]], 'k-', lw=1.5)
        poly = Polygon([B_start, V_start, V_end, B_end], facecolor='skyblue', edgecolor='dodgerblue', alpha=0.5)
        ax.add_patch(poly)

        ax.text(V_start[0] - 5, V_start[1], f"{sigma_max_sym}", color='red', fontsize=14, ha='right', va='center')
        ax.text(V_end[0] + 5, V_end[1], f"{sigma_min_sym}", color='red', fontsize=14, ha='left', va='center')

        ax.plot([self.ev(n_max.x, subs_dict), B_start[0]], [self.ev(n_max.y, subs_dict), B_start[1]], 'k--', lw=0.8)
        ax.plot([self.ev(n_min.x, subs_dict), B_end[0]], [self.ev(n_min.y, subs_dict), B_end[1]], 'k--', lw=0.8)

        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(r"$\sigma$ Normali (Navier)", fontsize=14)

    # ==========================================================
    # LOGICA GRAFICA 3: JOURAWSKY
    # ==========================================================
    def solve_jourawsky(self, loads, ax, subs_dict):
        Tx_tot = sum(l.value for l in loads if l.type == 'TX')
        Ty_tot = sum(l.value for l in loads if l.type == 'TY')
        
        if self.ev(Tx_tot, subs_dict) == 0 and self.ev(Ty_tot, subs_dict) == 0:
            ax.axis('off')
            ax.set_title("Jourawsky: Nessun carico di taglio", fontsize=12)
            return

        Delta = sym_ex(self.I_x * self.I_y - self.I_xy**2)
        if Delta == 0: return

        element_tau_sym = {}
        element_tau_num = {}
        max_tau_num_global = 0.0
        max_tau_sym_global = 0
        
        degree = {n_id: len(n.connected_elements) for n_id, n in self.nodes.items()}
        leaves = [n_id for n_id, deg in degree.items() if deg == 1]
        
        node_Sx = {n_id: 0 for n_id in self.nodes}
        node_Sy = {n_id: 0 for n_id in self.nodes}
        processed_elements = set()
        
        print(f"=== RISULTATI JOURAWSKY (Taglio) ===")
        print(f"Tx totale = {sym_ex(Tx_tot)} | Ty totale = {sym_ex(Ty_tot)}")

        zeta = sp.Symbol('zeta', real=True)

        # FASE 1: LEAF REMOVAL
        while leaves:
            curr_node_id = leaves.pop(0)
            curr_node = self.nodes[curr_node_id]
            
            unprocessed_elems = [e for e in curr_node.connected_elements if e.id not in processed_elements]
            if not unprocessed_elems: continue
                
            el = unprocessed_elems[0]
            processed_elements.add(el.id)
            next_node = el.n1 if el.n2 == curr_node else el.n2
            
            x1 = sym_ex(curr_node.x - self.x_G)
            y1 = sym_ex(curr_node.y - self.y_G)
            x2 = sym_ex(next_node.x - self.x_G)
            y2 = sym_ex(next_node.y - self.y_G)
            
            S_x_start, S_y_start = node_Sx[curr_node_id], node_Sy[curr_node_id]
            
            area_z = sym_ex(el.area * zeta)
            cx_z = sym_ex(x1 + zeta * (x2 - x1) * sp.Rational(1,2))
            cy_z = sym_ex(y1 + zeta * (y2 - y1) * sp.Rational(1,2))
            
            Sx_z = sym_ex(S_x_start + area_z * cy_z)
            Sy_z = sym_ex(S_y_start + area_z * cx_z)
            
            q_z = sym_ex((Tx_tot * (self.I_x * Sy_z - self.I_xy * Sx_z) + 
                          Ty_tot * (self.I_y * Sx_z - self.I_xy * Sy_z)) / Delta)
                               
            tau_sym = sym_ex(q_z / el.t)
            element_tau_sym[el.id] = (tau_sym, curr_node, next_node)

            num_points = 21
            z_vals = np.linspace(0, 1, num_points)
            tau_vals_num = np.array([self.ev(tau_sym.subs(zeta, z), subs_dict) for z in z_vals])
            
            local_max_num = np.max(np.abs(tau_vals_num))
            if local_max_num > max_tau_num_global:
                max_tau_num_global = local_max_num
                idx_max = np.argmax(np.abs(tau_vals_num))
                max_tau_sym_global = sym_ex(tau_sym.subs(zeta, sp.Rational(idx_max, num_points-1)))
            
            flow_dir_x = self.ev(next_node.x, subs_dict) - self.ev(curr_node.x, subs_dict)
            flow_dir_y = self.ev(next_node.y, subs_dict) - self.ev(curr_node.y, subs_dict)
            L_num = np.hypot(flow_dir_x, flow_dir_y)
            flow_dir = np.array([flow_dir_x/L_num, flow_dir_y/L_num])
            
            pts_x = self.ev(curr_node.x, subs_dict) + z_vals * flow_dir_x
            pts_y = self.ev(curr_node.y, subs_dict) + z_vals * flow_dir_y
            element_tau_num[el.id] = [pts_x, pts_y, tau_vals_num, flow_dir, tau_sym]
            
            node_Sx[next_node.id] += sym_ex(S_x_start + el.area * sp.Rational(1,2) * (y1 + y2))
            node_Sy[next_node.id] += sym_ex(S_y_start + el.area * sp.Rational(1,2) * (x1 + x2))
            
            degree[next_node.id] -= 1
            if degree[next_node.id] == 1:
                leaves.append(next_node.id)

        # FASE 2: SEZIONE CHIUSA
        unprocessed_elems = [el for el in self.elements if el.id not in processed_elements]
        if unprocessed_elems:
            ring_path = [] 
            curr_node = unprocessed_elems[0].n1
            visited_ring_elems = set()
            while len(visited_ring_elems) < len(unprocessed_elems):
                next_el = None
                for el in curr_node.connected_elements:
                    if el in unprocessed_elems and el.id not in visited_ring_elems:
                        next_el = el
                        break
                if not next_el: break
                next_node = next_el.n2 if next_el.n1 == curr_node else next_el.n1
                ring_path.append((next_el, curr_node, next_node))
                visited_ring_elems.add(next_el.id)
                curr_node = next_node

            current_Sx, current_Sy = 0, 0
            integral_q_ds, integral_ds_t = 0, 0
            ring_tau_data = {}
            
            for el, n_start, n_end in ring_path:
                current_Sx += node_Sx[n_start.id]
                current_Sy += node_Sy[n_start.id]
                node_Sx[n_start.id], node_Sy[n_start.id] = 0, 0 
                
                x1 = sym_ex(n_start.x - self.x_G)
                y1 = sym_ex(n_start.y - self.y_G)
                x2 = sym_ex(n_end.x - self.x_G)
                y2 = sym_ex(n_end.y - self.y_G)
                
                area_z = sym_ex(el.area * zeta)
                cx_z = sym_ex(x1 + zeta * (x2 - x1) * sp.Rational(1,2))
                cy_z = sym_ex(y1 + zeta * (y2 - y1) * sp.Rational(1,2))
                Sx_z = sym_ex(current_Sx + area_z * cy_z)
                Sy_z = sym_ex(current_Sy + area_z * cx_z)
                
                q_z = sym_ex((Tx_tot * (self.I_x * Sy_z - self.I_xy * Sx_z) + 
                              Ty_tot * (self.I_y * Sx_z - self.I_xy * Sy_z)) / Delta)
                tau_sym = sym_ex(q_z / el.t)
                ring_tau_data[el.id] = (tau_sym, n_start, n_end)
                
                q_in = sym_ex(tau_sym.subs(zeta, 0) * el.t)
                q_mid = sym_ex(tau_sym.subs(zeta, sp.Rational(1,2)) * el.t)
                q_out = sym_ex(tau_sym.subs(zeta, 1) * el.t)
                
                integral_q_ds += sym_ex((el.length * sp.Rational(1,6)) * (q_in + 4*q_mid + q_out) / el.t)
                integral_ds_t += sym_ex(el.length / el.t)
                
                current_Sx += sym_ex(el.area * sp.Rational(1,2) * (y1 + y2))
                current_Sy += sym_ex(el.area * sp.Rational(1,2) * (x1 + x2))
                
            q0 = sym_ex(- integral_q_ds / integral_ds_t)
            
            for el, n_start, n_end in ring_path:
                tau_sym, n_start, n_end = ring_tau_data[el.id]
                tau_corrected_sym = sym_ex(tau_sym + (q0 / el.t))
                
                num_points = 21
                z_vals = np.linspace(0, 1, num_points)
                tau_vals_num = np.array([self.ev(tau_corrected_sym.subs(zeta, z), subs_dict) for z in z_vals])
                
                local_max_num = np.max(np.abs(tau_vals_num))
                if local_max_num > max_tau_num_global:
                    max_tau_num_global = local_max_num
                    idx_max = np.argmax(np.abs(tau_vals_num))
                    max_tau_sym_global = sym_ex(tau_corrected_sym.subs(zeta, sp.Rational(idx_max, num_points-1)))

                flow_dir_x = self.ev(n_end.x, subs_dict) - self.ev(n_start.x, subs_dict)
                flow_dir_y = self.ev(n_end.y, subs_dict) - self.ev(n_start.y, subs_dict)
                L_num = np.hypot(flow_dir_x, flow_dir_y)
                flow_dir = np.array([flow_dir_x/L_num, flow_dir_y/L_num])
                
                pts_x = self.ev(n_start.x, subs_dict) + z_vals * flow_dir_x
                pts_y = self.ev(n_start.y, subs_dict) + z_vals * flow_dir_y
                element_tau_num[el.id] = [pts_x, pts_y, tau_vals_num, flow_dir, tau_corrected_sym]

        print(f"Massima Tensione Tangenziale: {max_tau_sym_global}\n")

        # PLOT
        x_vals = [self.ev(n.x, subs_dict) for n in self.nodes.values()]
        w = max(x_vals) - min(x_vals) if x_vals else 100
        scale = (w * 0.15) / max_tau_num_global if max_tau_num_global > 0 else 1.0

        for el in self.elements:
            ax.plot([self.ev(el.n1.x, subs_dict), self.ev(el.n2.x, subs_dict)], 
                    [self.ev(el.n1.y, subs_dict), self.ev(el.n2.y, subs_dict)], 'k-', linewidth=self.ev(el.t, subs_dict), solid_capstyle='round')
            
            if el.id in element_tau_num:
                x_pts, y_pts, taus_num, flow_dir, tau_sym_expr = element_tau_num[el.id]
                perp_dir = np.array([-flow_dir[1], flow_dir[0]])
                
                poly_pts = []
                for x, y in zip(x_pts, y_pts): poly_pts.append([x, y])
                for x, y, t in zip(reversed(x_pts), reversed(y_pts), reversed(taus_num)):
                    poly_pts.append([x + perp_dir[0] * t * scale, y + perp_dir[1] * t * scale])
                    
                poly = Polygon(poly_pts, facecolor='skyblue', edgecolor='dodgerblue', alpha=0.5)
                ax.add_patch(poly)
                
                num_tot_pts = len(taus_num)
                for idx in [int(num_tot_pts*0.25), int(num_tot_pts*0.75)]: 
                    if abs(taus_num[idx]) > 1e-4:
                        f_dir = flow_dir if taus_num[idx] < 0 else -flow_dir
                        f_x = x_pts[idx] + perp_dir[0] * (taus_num[idx] * scale) * 0.5
                        f_y = y_pts[idx] + perp_dir[1] * (taus_num[idx] * scale) * 0.5
                        ax.arrow(f_x, f_y, f_dir[0]*w*0.03, f_dir[1]*w*0.03, color='dodgerblue', head_width=w*0.015, zorder=3)

                max_idx = np.argmax(np.abs(taus_num))
                if abs(taus_num[max_idx]) > 1e-6:
                    max_sym_val = sym_ex(tau_sym_expr.subs(zeta, sp.Rational(max_idx, num_tot_pts-1)))
                    tx = x_pts[max_idx] + perp_dir[0] * taus_num[max_idx] * scale * 1.2
                    ty = y_pts[max_idx] + perp_dir[1] * taus_num[max_idx] * scale * 1.2
                    ax.text(tx, ty, f"{max_sym_val}", color='red', fontsize=12, ha='center', va='center')

        xg, yg = self.ev(self.x_G, subs_dict), self.ev(self.y_G, subs_dict)
        ax.plot(xg, yg, 'ko') 
        ax.text(xg + 2, yg - 2, 'G', fontsize=12)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(r"$\tau$ Tangenziali (Jourawsky)", fontsize=14)

    # ==========================================================
    # LOGICA GRAFICA 4: TORSIONE
    # ==========================================================
    def solve_torsion(self, loads, ax, subs_dict):
        Mt_tot = 0
        for load in loads:
            if load.type == 'MT':
                Mt_tot += sym_ex(load.value)
            elif load.type == 'TY' and load.x is not None:
                e_x = sym_ex(self.x_CT - load.x)
                Mt_tot += sym_ex(load.value * e_x)
            elif load.type == 'TX' and load.y is not None:
                e_y = sym_ex(self.y_CT - load.y)
                Mt_tot += sym_ex(-load.value * e_y)

        Mt_tot = sym_ex(Mt_tot)

        if abs(self.ev(Mt_tot, subs_dict)) < 1e-6:
            ax.axis('off')
            ax.set_title("Torsione: Nessun Momento Torcente rilevato", fontsize=12)
            return

        print(f"=== RISULTATI TORSIONE PURA ===")
        print(f"Momento Torcente Totale sul CT (Mt) = {Mt_tot}")

        cycle_nodes = self._find_fundamental_cycle()
        cycle_elems = self._get_elements_in_cycle(cycle_nodes) if cycle_nodes else []

        Jtc = 0
        omega = 0
        if cycle_nodes:
            for i in range(len(cycle_nodes)):
                n1 = cycle_nodes[i]
                n2 = cycle_nodes[(i + 1) % len(cycle_nodes)]
                omega += sym_ex(n1.x * n2.y - n2.x * n1.y)
            omega = sym_ex(abs(omega) * sp.Rational(1,2))
            integral_ds_t = sum(sym_ex(el.length / el.t) for el in cycle_elems)
            Jtc = sym_ex((4 * omega**2) / integral_ds_t)

        Jto = 0
        open_elems = [el for el in self.elements if el not in cycle_elems]
        for el in open_elems:
            Jto += sym_ex(sp.Rational(1,3) * el.length * (el.t**3))

        Jt_tot = sym_ex(Jtc + Jto)
        if self.ev(Jt_tot, subs_dict) == 0: return

        Mtc = sym_ex(Mt_tot * (Jtc / Jt_tot))
        Mto = sym_ex(Mt_tot * (Jto / Jt_tot))

        tau_results_sym = {}
        for el in self.elements:
            if el in cycle_elems:
                tau = sym_ex(Mtc / (2 * omega * el.t)) if self.ev(omega, subs_dict) > 0 else 0
                tau_results_sym[el.id] = {'type': 'closed', 'val': tau}
            else:
                tau = sym_ex((Mto / Jto) * el.t) if self.ev(Jto, subs_dict) > 0 else 0
                tau_results_sym[el.id] = {'type': 'open', 'val': tau}

        if cycle_nodes:
            print(f"Jt_chiusa (Bredt) = {Jtc} | Jt_aperta (Ali) = {Jto}")
            print(f"Mt cella chiusa = {Mtc} | Mt ali aperte = {Mto}")
        else:
            print(f"Jt_totale (Aperta) = {Jto}")

        for el_id, data in tau_results_sym.items():
            print(f"Elem {el_id} ({data['type']}): tau_max = {sym_ex(abs(data['val']))}")
        print("\n")

        x_vals = [self.ev(n.x, subs_dict) for n in self.nodes.values()]
        w = max(x_vals) - min(x_vals) if x_vals else 100

        max_tau_num = max(abs(self.ev(d['val'], subs_dict)) for d in tau_results_sym.values())
        scale_tau = (w * 0.25) / max_tau_num if max_tau_num > 0 else 1.0
        W_draw = w * 0.03 

        for el in self.elements:
            ax.plot([self.ev(el.n1.x, subs_dict), self.ev(el.n2.x, subs_dict)], 
                    [self.ev(el.n1.y, subs_dict), self.ev(el.n2.y, subs_dict)], color='k', linewidth=self.ev(el.t, subs_dict), solid_capstyle='round')

            tau_sym = tau_results_sym[el.id]['val']
            tau_num = self.ev(tau_sym, subs_dict)
            if abs(tau_num) < 1e-4: continue

            cx, cy = self.ev(el.center_of_gravity[0], subs_dict), self.ev(el.center_of_gravity[1], subs_dict)
            dx = self.ev(el.n2.x, subs_dict) - self.ev(el.n1.x, subs_dict)
            dy = self.ev(el.n2.y, subs_dict) - self.ev(el.n1.y, subs_dict)
            L = np.hypot(dx, dy)
            u = np.array([dx/L, dy/L]) 
            v = np.array([-u[1], u[0]]) 

            sign = 1 if self.ev(Mt_tot, subs_dict) < 0 else -1
            H_vec = u * abs(tau_num) * scale_tau * sign

            if tau_results_sym[el.id]['type'] == 'open':
                pA_base = np.array([cx, cy]) + v * W_draw
                pA_tip  = pA_base - H_vec
                
                pB_base = np.array([cx, cy]) - v * W_draw
                pB_tip  = pB_base + H_vec

                poly1 = Polygon([[cx, cy], pA_base, pA_tip], facecolor='skyblue', edgecolor='dodgerblue', alpha=0.5)
                poly2 = Polygon([[cx, cy], pB_base, pB_tip], facecolor='skyblue', edgecolor='dodgerblue', alpha=0.5)
                ax.add_patch(poly1)
                ax.add_patch(poly2)

                ax.plot([pB_base[0], pA_base[0]], [pB_base[1], pA_base[1]], 'k-', lw=1)
                ax.arrow(pA_base[0], pA_base[1], -H_vec[0]*0.9, -H_vec[1]*0.9, color='dodgerblue', head_width=W_draw*0.8, zorder=3)
                ax.arrow(pB_base[0], pB_base[1], H_vec[0]*0.9, H_vec[1]*0.9, color='dodgerblue', head_width=W_draw*0.8, zorder=3)
                ax.text(pA_tip[0] + u[0]*W_draw*2, pA_tip[1] + u[1]*W_draw*2, f"{sym_ex(abs(tau_sym))}", color='red', fontsize=12, ha='center', va='center')

            else:
                r = np.array([cx - self.ev(self.x_CT, subs_dict), cy - self.ev(self.y_CT, subs_dict)])
                flow_sign = 1 if (r[0]*u[1] - r[1]*u[0]) * self.ev(Mt_tot, subs_dict) > 0 else -1
                H_vec_c = u * abs(tau_num) * scale_tau * flow_sign

                pA_base = np.array([cx, cy]) + v * W_draw
                pB_base = np.array([cx, cy]) - v * W_draw
                pA_tip = pA_base + H_vec_c
                pB_tip = pB_base + H_vec_c

                poly = Polygon([pA_base, pB_base, pB_tip, pA_tip], facecolor='skyblue', edgecolor='dodgerblue', alpha=0.5)
                ax.add_patch(poly)
                ax.plot([pB_base[0], pA_base[0]], [pB_base[1], pA_base[1]], 'k-', lw=1)

                mid_base = np.array([cx, cy])
                ax.arrow(mid_base[0], mid_base[1], H_vec_c[0]*0.9, H_vec_c[1]*0.9, color='dodgerblue', head_width=W_draw*0.8, zorder=3)
                ax.text(pA_tip[0] + u[0]*W_draw*2, pA_tip[1] + u[1]*W_draw*2, f"{sym_ex(abs(tau_sym))}", color='red', fontsize=12, ha='center', va='center')

        ax.plot(self.ev(self.x_CT, subs_dict), self.ev(self.y_CT, subs_dict), 'g*', markersize=10, zorder=4)
        ax.plot(self.ev(self.x_G, subs_dict), self.ev(self.y_G, subs_dict), 'ko', markersize=6, zorder=4)

        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(r"$\tau$ Torsionali", fontsize=14)

    def solve_and_plot_all(self, loads, subs_dict):
        self.print_recap()
        fig, axs = plt.subplots(2, 2, figsize=(16, 14))
        self.plot_loads_summary(loads, axs[0, 0], subs_dict)
        self.solve_navier(loads, axs[0, 1], subs_dict)
        self.solve_jourawsky(loads, axs[1, 0], subs_dict)
        self.solve_torsion(loads, axs[1, 1], subs_dict)
        plt.tight_layout()
        plt.show()


# ==========================================================
# ESECUZIONE E TESTING DEGLI SCENARI
# ==========================================================
import streamlit as st
import pandas as pd

# ==========================================================
# INTERFACCIA WEB CON STREAMLIT
# ==========================================================
st.set_page_config(page_title="Analisi Sezioni", layout="wide")

st.title("📐 Analisi Tensionale Sezioni Sottili")
st.markdown("Definisci la geometria e i carichi usando espressioni simboliche (es. `b`, `t`, `2*b`).")

# --- COLONNE DI LAYOUT ---
col_input, col_output = st.columns([1, 2])

with col_input:
    st.header("1. Parametri Numerici")
    st.markdown("Definisci il valore reale dei simboli che userai.")
    
    # Tabella per i parametri
    df_params_init = pd.DataFrame([
        {"Simbolo": "b", "Valore": 100.0},
        {"Simbolo": "t", "Valore": 5.0},
        {"Simbolo": "N", "Valore": 10000.0},
        {"Simbolo": "T", "Valore": 20000.0}
    ])
    df_params = st.data_editor(df_params_init, num_rows="dynamic", use_container_width=True)

    st.header("2. Topologia")
    # Tabella Nodi
    st.subheader("Nodi")
    df_nodi_init = pd.DataFrame([
        {"ID": 1, "X": "b", "Y": "-b"},
        {"ID": 2, "X": "0", "Y": "-b"},
        {"ID": 3, "X": "0", "Y": "0"},
        {"ID": 4, "X": "4*b", "Y": "0"},
        {"ID": 5, "X": "4*b", "Y": "-b"},
        {"ID": 6, "X": "3*b", "Y": "-b"}
    ])
    df_nodi = st.data_editor(df_nodi_init, num_rows="dynamic", use_container_width=True)

    # Tabella Elementi
    st.subheader("Elementi")
    df_elem_init = pd.DataFrame([
        {"ID": 1, "Nodo 1": 1, "Nodo 2": 2, "Spessore": "2*t"},
        {"ID": 2, "Nodo 1": 2, "Nodo 2": 3, "Spessore": "t"},
        {"ID": 3, "Nodo 1": 3, "Nodo 2": 4, "Spessore": "t"},
        {"ID": 4, "Nodo 1": 4, "Nodo 2": 5, "Spessore": "t"},
        {"ID": 5, "Nodo 1": 5, "Nodo 2": 6, "Spessore": "2*t"}
    ])
    df_elem = st.data_editor(df_elem_init, num_rows="dynamic", use_container_width=True)

    st.header("3. Carichi Applicati")
    df_carichi_init = pd.DataFrame([
        {"Tipo (N, TX, TY, MT, MX, MY)": "N", "Valore": "N", "X_app": "4*b", "Y_app": "0"},
        {"Tipo (N, TX, TY, MT, MX, MY)": "MT", "Valore": "N*b", "X_app": "", "Y_app": ""}
    ])
    df_carichi = st.data_editor(df_carichi_init, num_rows="dynamic", use_container_width=True)

    esegui = st.button("🚀 Calcola e Disegna", type="primary", use_container_width=True)

with col_output:
    st.header("Risultati")
    
    if esegui:
        try:
            # 1. Costruisco il dizionario delle sostituzioni
            subs_dict = {}
            for _, row in df_params.iterrows():
                if row["Simbolo"]:
                    sym = sym_ex(row["Simbolo"])
                    subs_dict[sym] = float(row["Valore"])

            # 2. Costruisco i Nodi
            nodi_dict = {}
            for _, row in df_nodi.iterrows():
                nodi_dict[row["ID"]] = Node(int(row["ID"]), row["X"], row["Y"])

            # 3. Costruisco gli Elementi
            elementi = []
            for _, row in df_elem.iterrows():
                n1 = nodi_dict[row["Nodo 1"]]
                n2 = nodi_dict[row["Nodo 2"]]
                elementi.append(Element(int(row["ID"]), n1, n2, row["Spessore"]))

            # 4. Creo la Sezione
            sezione = CrossSection(list(nodi_dict.values()), elementi)

            # 5. Costruisco i Carichi
            carichi = []
            for _, row in df_carichi.iterrows():
                if row["Tipo (N, TX, TY, MT, MX, MY)"]:
                    x_app = row["X_app"] if pd.notna(row["X_app"]) and row["X_app"] != "" else None
                    y_app = row["Y_app"] if pd.notna(row["Y_app"]) and row["Y_app"] != "" else None
                    carichi.append(Load(row["Tipo (N, TX, TY, MT, MX, MY)"], row["Valore"], x_app, y_app))

            # 6. CATTURA DELL'OUTPUT E CALCOLO
            # Creiamo un "finto terminale" in memoria per catturare i print()
            finto_terminale = io.StringIO()
            
            with redirect_stdout(finto_terminale):
                sezione.print_recap() # <--- Aggiunto per avere il recap geometrico!
                fig, axs = plt.subplots(2, 2, figsize=(16, 14))
                sezione.plot_loads_summary(carichi, axs[0, 0], subs_dict)
                sezione.solve_navier(carichi, axs[0, 1], subs_dict)
                sezione.solve_jourawsky(carichi, axs[1, 0], subs_dict)
                sezione.solve_torsion(carichi, axs[1, 1], subs_dict)
                plt.tight_layout()

            # 7. MOSTRA I RISULTATI NELLA WEB APP USANDO I TABS
            tab_grafici, tab_testo = st.tabs(["📊 Grafici", "📄 Report Analitico"])
            
            with tab_grafici:
                st.pyplot(fig)
                
            with tab_testo:
                # Stampiamo il testo catturato usando un blocco di codice per mantenere l'impaginazione
                st.code(finto_terminale.getvalue(), language="text")

        except Exception as e:
            st.error(f"Si è verificato un errore durante il calcolo: {e}")
            st.info("Controlla che la topologia sia ben connessa e che le formule scritte nelle tabelle siano corrette (es. usa '*' per le moltiplicazioni come '2*b').")
    else:
        st.info("Compila i dati a sinistra e clicca su 'Calcola e Disegna'.")
