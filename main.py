import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mealpy import FloatVar, Problem, ES
from typing import Dict, List, Any
import os
from mpl_toolkits.mplot3d import Axes3D

class EnhancedSpringOptimizer:
    def __init__(self):
        # Configurações de otimização
        self.max_epoch = 800  # Balanceamento entre tempo e qualidade
        self.pop_size = 120
        self.n_runs = 5  # Número de execuções por estratégia
        
        # Referência do artigo (Harris Hawks Optimization)
        self.hho_ref = {
            'd': 0.051796393,
            'D': 0.359305355,
            'N': 11.138859,
            'weight': 0.012665443
        }
        
        # Espaço de busca
        self.bounds = FloatVar(lb=[0.05, 0.25, 2.0], ub=[2.0, 1.3, 15.0])

    def calculate_constraints(self, solution: np.ndarray) -> List[float]:
        """Calcula todas as restrições com proteção numérica"""
        d, D, N = solution
        eps = 1e-20
        return [
            max(0, 1 - (D**3 * N)/(71785 * d**4 + eps)),
            max(0, (4*D**2 - d*D)/(12566*(D*d**3 - d**4 + eps)) + 1/(5108*d**2) - 1),
            max(0, 1 - (140.45*d)/(D**2 * N + eps)),
            max(0, (d + D)/1.5 - 1)
        ]

    def dynamic_penalty(self, violations: List[float], epoch: int, strategy: str) -> float:
        """Implementa todas as estratégias de penalização"""
        sum_viol = sum(v**2 for v in violations)
        max_viol = max(violations)
        progress = epoch / self.max_epoch
        
        if strategy == "progressive":
            return (1e5 + 1e7 * progress) * sum_viol
        elif strategy == "violation":
            return 1e6 * (sum_viol + 100 * max_viol)
        elif strategy == "hybrid":
            return (1e5 + 5e6 * progress) * (sum_viol + 50 * max_viol * progress)
        elif strategy == "no-penalty":
            return 0
        else:  # fixed
            return 1e5 * sum_viol

    def evaluate(self, solution: np.ndarray, epoch: int, strategy: str) -> float:
        """Função objetivo completa"""
        weight = (solution[2] + 2) * solution[1] * solution[0]**2
        violations = self.calculate_constraints(solution)
        penalty = self.dynamic_penalty(violations, epoch, strategy)
        return weight + penalty

    def run_strategy(self, strategy: str) -> Dict[str, Any]:
        """Executa uma estratégia específica"""
        class CustomES(ES.OriginalES):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.strategy = strategy
            
            def evolve(self, epoch: int):
                self.problem.obj_func = lambda x: self.optimizer.evaluate(x, epoch, self.strategy)
                super().evolve(epoch)
        
        model = CustomES(epoch=self.max_epoch, pop_size=self.pop_size)
        model.optimizer = self
        
        problem = Problem(
            bounds=self.bounds,
            minmax="min",
            obj_func=lambda x: self.evaluate(x, 0, strategy)
        )
        
        model.solve(problem)
        solution = model.g_best.solution
        violations = self.calculate_constraints(solution)
        
        return {
            'strategy': strategy,
            'd': solution[0],
            'D': solution[1],
            'N': solution[2],
            'weight': (solution[2] + 2) * solution[1] * solution[0]**2,
            'violations': violations,
            'feasible': all(v <= 1e-6 for v in violations),
            'history': model.history.list_global_best_fit,
            'time': sum(model.history.list_epoch_time)
        }

    def comprehensive_analysis(self):
        """Executa todas as análises requeridas"""
        os.makedirs('results', exist_ok=True)
        
        # 1. Testar todas as estratégias
        strategies = ['fixed', 'progressive', 'violation', 'hybrid', 'no-penalty']
        all_results = []
        
        for strategy in strategies:
            print(f"\n⚡ Executando estratégia: {strategy.upper()}")
            strategy_results = []
            
            for run in range(self.n_runs):
                res = self.run_strategy(strategy)
                strategy_results.append(res)
                print(f"Execução {run+1}: Peso = {res['weight']:.6f} kg | Viável = {res['feasible']}")
            
            # Salvar resultados por estratégia
            pd.DataFrame(strategy_results).to_csv(f'results/{strategy}_results.csv', index=False)
            all_results.extend(strategy_results)
        
        # 2. Processar resultados
        df = pd.DataFrame(all_results)
        
        # 3. Gerar relatórios
        self.generate_reports(df)
        
        return df

    def generate_reports(self, df: pd.DataFrame):
        """Gera todos os relatórios e gráficos"""
        # 1. Melhores resultados por estratégia
        best_results = df.loc[df.groupby('strategy')['weight'].idxmin()]
        print("\n⭐ MELHORES RESULTADOS POR ESTRATÉGIA:")
        print(best_results[['strategy', 'd', 'D', 'N', 'weight', 'feasible']].to_markdown(index=False))
        
        # 2. Comparação com HHO
        comparison = pd.DataFrame([
            {
                'Approach': 'HHO (Artigo)',
                'd': self.hho_ref['d'],
                'D': self.hho_ref['D'],
                'N': self.hho_ref['N'],
                'Weight': self.hho_ref['weight'],
                'Feasible': True
            },
            *[{
                'Approach': f"ES ({row['strategy']})", 
                'd': row['d'],
                'D': row['D'],
                'N': row['N'],
                'Weight': row['weight'],
                'Feasible': row['feasible']
            } for _, row in best_results.iterrows()]
        ])
        comparison.to_csv('results/final_comparison.csv', index=False)
        
        # 3. Gráfico de convergência
        plt.figure(figsize=(12,6))
        for strategy in df['strategy'].unique():
            subset = df[df['strategy'] == strategy]
            plt.plot(subset['history'].iloc[0], label=f"{strategy} ({subset['weight'].min():.6f} kg)")
        plt.axhline(y=self.hho_ref['weight'], color='black', linestyle='--', label='HHO Reference')
        plt.legend()
        plt.savefig('results/convergence_comparison.png', dpi=300)
        
        # 4. Visualização 3D
        fig = plt.figure(figsize=(14,8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot HHO reference
        ax.scatter(self.hho_ref['d'], self.hho_ref['D'], self.hho_ref['N'], 
                  c='red', s=100, label='HHO (Artigo)')
        
        # Plot our results
        for _, row in best_results.iterrows():
            ax.scatter(row['d'], row['D'], row['N'], 
                      label=f"{row['strategy']} ({row['weight']:.6f} kg)")
        
        ax.set_xlabel('d (diâmetro)')
        ax.set_ylabel('D (bobina)')
        ax.set_zlabel('N (espiras)')
        plt.legend()
        plt.savefig('results/3d_solutions.png', dpi=300)
        
        print("\n✅ Relatórios gerados na pasta /results:")
        print("- final_comparison.csv")
        print("- convergence_comparison.png")
        print("- 3d_solutions.png")

if __name__ == "__main__":
    optimizer = EnhancedSpringOptimizer()
    results = optimizer.comprehensive_analysis()
