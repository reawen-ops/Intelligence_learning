import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Callable


class GeneticAlgorithmTSP:
    def __init__(self, cities: List[Tuple[float, float]], population_size: int = 100,
                 generations: int = 500, mutation_rate: float = 0.02,
                 tournament_size: int = 5, elitism_count: int = 2):
        """
        初始化遗传算法TSP求解器

        参数:
        cities: 城市坐标列表，格式为[(x1, y1), (x2, y2), ...]
        population_size: 种群大小
        generations: 进化代数
        mutation_rate: 变异率
        tournament_size: 锦标赛选择的大小
        elitism_count: 精英保留数量
        """
        self.cities = cities
        self.num_cities = len(cities)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count

        # 预计算城市之间的距离矩阵
        self.distance_matrix = self._calculate_distance_matrix()

    def _calculate_distance_matrix(self) -> np.ndarray:
        """计算城市之间的距离矩阵"""
        n = self.num_cities
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # 计算欧几里得距离
                dist = np.sqrt((self.cities[i][0] - self.cities[j][0]) ** 2 +
                               (self.cities[i][1] - self.cities[j][1]) ** 2)
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist

        return dist_matrix

    def _calculate_route_distance(self, route: List[int]) -> float:
        """计算路径的总距离"""
        total_distance = 0
        for i in range(self.num_cities):
            from_city = route[i]
            to_city = route[(i + 1) % self.num_cities]  # 回到起点
            total_distance += self.distance_matrix[from_city][to_city]
        return total_distance

    def _create_individual(self) -> List[int]:
        """创建一个随机的个体（路径）"""
        individual = list(range(self.num_cities))
        random.shuffle(individual)
        return individual

    def _initialize_population(self) -> List[List[int]]:
        """初始化种群"""
        return [self._create_individual() for _ in range(self.population_size)]

    def _fitness(self, individual: List[int]) -> float:
        """计算适应度（路径距离的倒数，距离越短适应度越高）"""
        distance = self._calculate_route_distance(individual)
        return 1.0 / distance  # 距离越小，适应度越高

    def _tournament_selection(self, population: List[List[int]],
                              fitnesses: List[float]) -> List[int]:
        """锦标赛选择"""
        # 随机选择 tournament_size 个个体
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

        # 选择适应度最高的个体
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index]

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """顺序交叉（OX）"""
        size = len(parent1)

        # 选择两个交叉点
        cx1, cx2 = sorted(random.sample(range(size), 2))

        # 初始化子代
        child = [-1] * size

        # 从parent1复制交叉点之间的片段
        child[cx1:cx2] = parent1[cx1:cx2]

        # 从parent2填充剩余位置
        parent2_pos = 0
        for i in range(size):
            if child[i] == -1:  # 需要填充的位置
                while parent2[parent2_pos] in child:
                    parent2_pos += 1
                child[i] = parent2[parent2_pos]
                parent2_pos += 1

        return child

    def _swap_mutation(self, individual: List[int]) -> List[int]:
        """交换变异：随机交换两个城市的位置"""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def _inversion_mutation(self, individual: List[int]) -> List[int]:
        """倒置变异：随机选择一段路径并反转"""
        if random.random() < self.mutation_rate:
            start, end = sorted(random.sample(range(len(individual)), 2))
            individual[start:end + 1] = reversed(individual[start:end + 1])
        return individual

    def evolve(self) -> Tuple[List[int], float, List[float]]:
        """执行遗传算法进化过程"""
        # 初始化种群
        population = self._initialize_population()
        best_fitness_history = []
        avg_fitness_history = []

        best_individual = None
        best_fitness = 0

        for generation in range(self.generations):
            # 计算适应度
            fitnesses = [self._fitness(ind) for ind in population]

            # 记录最佳个体
            current_best_fitness = max(fitnesses)
            current_best_index = np.argmax(fitnesses)

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_index].copy()

            # 记录历史数据
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(np.mean(fitnesses))

            # 创建新种群
            new_population = []

            # 精英保留
            elite_indices = np.argsort(fitnesses)[-self.elitism_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # 生成剩余个体
            while len(new_population) < self.population_size:
                # 选择
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)

                # 交叉
                child = self._order_crossover(parent1, parent2)

                # 变异
                if random.random() < 0.5:
                    child = self._swap_mutation(child)
                else:
                    child = self._inversion_mutation(child)

                new_population.append(child)

            population = new_population

            # 每100代打印进度
            if generation % 100 == 0:
                best_distance = 1.0 / best_fitness
                print(f"Generation {generation}: Best Distance = {best_distance:.2f}")

        # 最终结果
        best_distance = 1.0 / best_fitness
        return best_individual, best_distance, best_fitness_history, avg_fitness_history

    def plot_results(self, best_route: List[int],
                     best_fitness_history: List[float],
                     avg_fitness_history: List[float]):
        """绘制结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 绘制最佳路径
        ax1.set_title('Best TSP Route')
        # 提取城市坐标
        x = [self.cities[i][0] for i in best_route] + [self.cities[best_route[0]][0]]
        y = [self.cities[i][1] for i in best_route] + [self.cities[best_route[0]][1]]

        ax1.plot(x, y, 'o-', markersize=8, linewidth=2)
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')

        # 标记城市编号
        for i, city in enumerate(best_route):
            ax1.annotate(str(city), (self.cities[city][0], self.cities[city][1]),
                         xytext=(5, 5), textcoords='offset points')

        # 绘制适应度进化曲线
        ax2.set_title('Fitness Evolution')
        ax2.plot(best_fitness_history, label='Best Fitness', linewidth=2)
        ax2.plot(avg_fitness_history, label='Average Fitness', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def generate_random_cities(num_cities: int, x_range: Tuple[float, float] = (0, 100),
                           y_range: Tuple[float, float] = (0, 100)) -> List[Tuple[float, float]]:
    """生成随机城市坐标"""
    return [(random.uniform(x_range[0], x_range[1]),
             random.uniform(y_range[0], y_range[1])) for _ in range(num_cities)]


# 示例使用
if __name__ == "__main__":
    # 设置随机种子以便重现结果
    random.seed(42)
    np.random.seed(42)

    # 生成20个随机城市
    num_cities = 20
    cities = generate_random_cities(num_cities)

    print(f"Generated {num_cities} random cities")
    print("Running Genetic Algorithm for TSP...")

    # 创建遗传算法求解器
    ga_tsp = GeneticAlgorithmTSP(
        cities=cities,
        population_size=100,
        generations=1000,
        mutation_rate=0.03,
        tournament_size=5,
        elitism_count=2
    )

    # 运行遗传算法
    best_route, best_distance, best_fitness_history, avg_fitness_history = ga_tsp.evolve()

    # 输出结果
    print(f"\n=== Results ===")
    print(f"Best route: {best_route}")
    print(f"Best distance: {best_distance:.2f}")
    print(f"Best route distance verification: {ga_tsp._calculate_route_distance(best_route):.2f}")

    # 绘制结果
    ga_tsp.plot_results(best_route, best_fitness_history, avg_fitness_history)