import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.padding import Padding
from rich.style import Style

# Helper function to calculate Highest Density Interval (HDI)
def _calculate_hdi(samples, credible_mass=0.95):
    """Calculate the Highest Density Interval (HDI) for a list of samples."""
    if samples is None or len(samples) == 0:
        return (np.nan, np.nan)
    
    samples = samples[~np.isnan(samples)] 
    if len(samples) == 0:
        return (np.nan, np.nan)
        
    sorted_samples = np.sort(samples)
    n_samples = len(samples)
    
    interval_idx_inc = int(np.floor(credible_mass * n_samples))
    if interval_idx_inc == 0: 
        return (np.nan, np.nan)
        
    n_intervals = n_samples - interval_idx_inc
    if n_intervals <= 0: 
         return (sorted_samples[0], sorted_samples[-1])

    interval_width = sorted_samples[interval_idx_inc:] - sorted_samples[:n_intervals]
    
    if len(interval_width) == 0:
        return (sorted_samples[0], sorted_samples[-1])
        
    min_idx = np.argmin(interval_width)
    hdi_min = sorted_samples[min_idx]
    hdi_max = sorted_samples[min_idx + interval_idx_inc]
    return hdi_min, hdi_max

class BayesianExperiment:
    """
    A class to perform Bayesian analysis for an A/B test with binomial data,
    supporting multiple solution variants.
    """
    def __init__(self, num_solution_variants=1):
        if not isinstance(num_solution_variants, int) or num_solution_variants < 1:
            raise ValueError("Number of solution variants must be a positive integer.")
        self.num_solution_variants = num_solution_variants

        # Control parameters
        self.control_prior_alpha = 1.0
        self.control_prior_beta = 1.0
        self.control_posterior_alpha = 1.0
        self.control_posterior_beta = 1.0
        self.control_samples = 0
        self.control_conversions = 0
        self.control_observed_alpha_likelihood = 1
        self.control_observed_beta_likelihood = 1

        # Solution variant parameters (lists)
        self.solution_prior_alpha = [1.0] * num_solution_variants
        self.solution_prior_beta = [1.0] * num_solution_variants
        self.solution_posterior_alpha = [1.0] * num_solution_variants
        self.solution_posterior_beta = [1.0] * num_solution_variants
        self.solution_samples = [0] * num_solution_variants
        self.solution_conversions = [0] * num_solution_variants
        self.solution_observed_alpha_likelihood = [1] * num_solution_variants
        self.solution_observed_beta_likelihood = [1] * num_solution_variants
        
        self.variant_names = ["Control"] + [f"Solution {i+1}" for i in range(num_solution_variants)]


    def set_priors(self, control_alpha, control_beta, solution_alphas, solution_betas):
        if control_alpha <= 0 or control_beta <= 0:
            raise ValueError("Control prior alpha and beta parameters must be positive.")
        if not (isinstance(solution_alphas, list) and isinstance(solution_betas, list) and
                len(solution_alphas) == self.num_solution_variants and len(solution_betas) == self.num_solution_variants):
            raise ValueError(f"Solution priors must be lists of length {self.num_solution_variants}.")
        for sa, sb in zip(solution_alphas, solution_betas):
            if sa <= 0 or sb <= 0:
                raise ValueError("Solution prior alpha and beta parameters must be positive.")

        self.control_prior_alpha = control_alpha
        self.control_prior_beta = control_beta
        self.control_posterior_alpha = control_alpha 
        self.control_posterior_beta = control_beta

        self.solution_prior_alpha = list(solution_alphas)
        self.solution_prior_beta = list(solution_betas)
        self.solution_posterior_alpha = list(solution_alphas) 
        self.solution_posterior_beta = list(solution_betas)


    def update_results(self, control_samples, control_conversions, solution_samples_list, solution_conversions_list):
        if control_samples < control_conversions or control_samples < 0:
            raise ValueError("Control samples must be non-negative and >= control conversions.")
        if not (isinstance(solution_samples_list, list) and isinstance(solution_conversions_list, list) and
                len(solution_samples_list) == self.num_solution_variants and len(solution_conversions_list) == self.num_solution_variants):
            raise ValueError(f"Solution results must be lists of length {self.num_solution_variants}.")

        self.control_samples = control_samples
        self.control_conversions = control_conversions
        control_losses = control_samples - control_conversions
        self.control_posterior_alpha = self.control_prior_alpha + control_conversions
        self.control_posterior_beta = self.control_prior_beta + control_losses
        self.control_observed_alpha_likelihood = control_conversions + (1 if control_conversions == 0 and control_losses == 0 else 0)
        self.control_observed_beta_likelihood = control_losses + (1 if control_conversions == 0 and control_losses == 0 else 0)

        for i in range(self.num_solution_variants):
            s_samples = solution_samples_list[i]
            s_conversions = solution_conversions_list[i]
            if s_samples < s_conversions or s_samples < 0:
                raise ValueError(f"Solution {i+1} samples must be non-negative and >= conversions.")
            if s_conversions < 0:
                raise ValueError(f"Solution {i+1} conversions must be non-negative.")
            
            self.solution_samples[i] = s_samples
            self.solution_conversions[i] = s_conversions
            s_losses = s_samples - s_conversions
            self.solution_posterior_alpha[i] = self.solution_prior_alpha[i] + s_conversions
            self.solution_posterior_beta[i] = self.solution_prior_beta[i] + s_losses
            self.solution_observed_alpha_likelihood[i] = s_conversions + (1 if s_conversions == 0 and s_losses == 0 else 0)
            self.solution_observed_beta_likelihood[i] = s_losses + (1 if s_conversions == 0 and s_losses == 0 else 0)


    def get_posterior_samples(self, n_samples=20000):
        """Generates posterior samples for control and all solution variants."""
        all_samples = {}
        all_samples['control_rate'] = stats.beta.rvs(
            self.control_posterior_alpha, self.control_posterior_beta, size=n_samples
        )
        for i in range(self.num_solution_variants):
            variant_name = f"solution_{i+1}_rate"
            all_samples[variant_name] = stats.beta.rvs(
                self.solution_posterior_alpha[i], self.solution_posterior_beta[i], size=n_samples
            )
        
        for i in range(self.num_solution_variants):
            all_samples[f"abs_diff_s{i+1}_c"] = all_samples[f"solution_{i+1}_rate"] - all_samples['control_rate']
        
        for i in range(self.num_solution_variants):
            rel_lift_samples = np.full_like(all_samples['control_rate'], np.nan)
            valid_mask = all_samples['control_rate'] > 1e-9
            rel_lift_samples[valid_mask] = (all_samples[f"solution_{i+1}_rate"][valid_mask] - all_samples['control_rate'][valid_mask]) / all_samples['control_rate'][valid_mask]
            all_samples[f"rel_lift_s{i+1}_c"] = rel_lift_samples
            
        return all_samples

    def calculate_metrics(self, rope_abs_diff=(-0.005, 0.005), rope_rel_lift=(-0.05, 0.05), 
                          prob_beat_threshold=0.0, credible_mass=0.95, n_samples_for_calc=20000):
        """Calculates metrics for control and all solution variants."""
        samples = self.get_posterior_samples(n_samples=n_samples_for_calc)
        metrics = {'control': {}, 'solutions': [{} for _ in range(self.num_solution_variants)]}

        control_s = samples['control_rate']
        metrics['control']['posterior_mean_rate'] = np.mean(control_s)
        metrics['control']['rate_hdi'] = _calculate_hdi(control_s, credible_mass)

        all_variant_samples = [samples['control_rate']] 
        for i in range(self.num_solution_variants):
            sol_s = samples[f"solution_{i+1}_rate"]
            abs_diff_s = samples[f"abs_diff_s{i+1}_c"]
            rel_lift_s = samples[f"rel_lift_s{i+1}_c"][~np.isnan(samples[f"rel_lift_s{i+1}_c"])]
            
            metrics['solutions'][i]['name'] = f"Solution {i+1}"
            metrics['solutions'][i]['posterior_mean_rate'] = np.mean(sol_s)
            metrics['solutions'][i]['rate_hdi'] = _calculate_hdi(sol_s, credible_mass)
            
            metrics['solutions'][i]['absolute_difference_mean'] = np.mean(abs_diff_s)
            metrics['solutions'][i]['absolute_difference_hdi'] = _calculate_hdi(abs_diff_s, credible_mass)
            
            if len(rel_lift_s) > 0:
                metrics['solutions'][i]['relative_lift_mean'] = np.mean(rel_lift_s)
                metrics['solutions'][i]['relative_lift_hdi'] = _calculate_hdi(rel_lift_s, credible_mass)
            else:
                metrics['solutions'][i]['relative_lift_mean'] = np.nan
                metrics['solutions'][i]['relative_lift_hdi'] = (np.nan, np.nan)

            metrics['solutions'][i]['prob_beats_control'] = np.mean(sol_s > control_s)
            metrics['solutions'][i]['prob_beats_control_by_threshold'] = np.mean(sol_s > (control_s + prob_beat_threshold))
            
            metrics['solutions'][i]['prob_abs_diff_below_rope'] = np.mean(abs_diff_s < rope_abs_diff[0])
            metrics['solutions'][i]['prob_abs_diff_in_rope'] = np.mean((abs_diff_s >= rope_abs_diff[0]) & (abs_diff_s <= rope_abs_diff[1]))
            metrics['solutions'][i]['prob_abs_diff_above_rope'] = np.mean(abs_diff_s > rope_abs_diff[1])
            
            if len(rel_lift_s) > 0:
                metrics['solutions'][i]['prob_rel_lift_below_rope'] = np.mean(rel_lift_s < rope_rel_lift[0])
                metrics['solutions'][i]['prob_rel_lift_in_rope'] = np.mean((rel_lift_s >= rope_rel_lift[0]) & (rel_lift_s <= rope_rel_lift[1]))
                metrics['solutions'][i]['prob_rel_lift_above_rope'] = np.mean(rel_lift_s > rope_rel_lift[1])
            else: 
                for key in ['prob_rel_lift_below_rope', 'prob_rel_lift_in_rope', 'prob_rel_lift_above_rope']:
                    metrics['solutions'][i][key] = np.nan

            metrics['solutions'][i]['expected_loss_vs_control_choosing_solution'] = np.mean(np.maximum(0, control_s - sol_s))
            metrics['solutions'][i]['expected_loss_vs_control_choosing_control'] = np.mean(np.maximum(0, sol_s - control_s))
            
            all_variant_samples.append(sol_s)

        stacked_samples = np.stack(all_variant_samples, axis=-1) 
        best_variant_indices = np.argmax(stacked_samples, axis=1)
        
        metrics['prob_control_is_best'] = np.mean(best_variant_indices == 0)
        for i in range(self.num_solution_variants):
            metrics['solutions'][i]['prob_is_best'] = np.mean(best_variant_indices == (i + 1))
            
        return metrics

    def get_decision_summary(self, metrics, rope_abs_diff_vs_control, p_threshold=0.95, loss_ratio_threshold=5):
        """Generates a decision summary for multiple variants."""
        best_overall_prob = metrics.get('prob_control_is_best', 0.0)
        best_variant_idx = -1 
        
        for i, sol_metrics in enumerate(metrics['solutions']):
            if sol_metrics.get('prob_is_best', 0.0) > best_overall_prob:
                best_overall_prob = sol_metrics.get('prob_is_best', 0.0)
                best_variant_idx = i
        
        if best_variant_idx == -1: 
            evaluation = "Control is Most Likely Best"
            recommendation = "Stick with Control"
            rec_style = "blue"
            for i, sol_metrics in enumerate(metrics['solutions']):
                 prob_s_beats_c = sol_metrics.get('prob_beats_control', 0.0)
                 loss_ctrl_vs_sol = sol_metrics.get('expected_loss_vs_control_choosing_control', np.inf)
                 loss_sol_vs_ctrl = sol_metrics.get('expected_loss_vs_control_choosing_solution', np.inf)
                 if prob_s_beats_c > 0.90 and loss_ctrl_vs_sol > loss_sol_vs_ctrl * (loss_ratio_threshold / 2): 
                      recommendation += f" (Consider Solution {i+1} if P(Best) for Control is marginal and risk is acceptable)"
                      break
            return evaluation, recommendation, rec_style

        best_sol_metrics = metrics['solutions'][best_variant_idx]
        evaluation = f"Solution {best_variant_idx+1} is Most Likely Best (P(Best)={best_sol_metrics.get('prob_is_best',0):.1%})"
        
        hdi_low, hdi_high = best_sol_metrics.get('absolute_difference_hdi', (np.nan, np.nan))
        rope_low, rope_high = rope_abs_diff_vs_control 
        prob_s_beats_c = best_sol_metrics.get('prob_beats_control', 0.0)
        loss_ctrl_vs_sol = best_sol_metrics.get('expected_loss_vs_control_choosing_control', np.inf)
        loss_sol_vs_ctrl = best_sol_metrics.get('expected_loss_vs_control_choosing_solution', np.inf)

        if np.isnan(hdi_low) or np.isnan(hdi_high):
            return evaluation, f"Error calculating HDI for Solution {best_variant_idx+1}", "red"

        if hdi_low > rope_high:
            recommendation = f"Accept Solution {best_variant_idx+1} (Clear Win vs Control)"
            rec_style = "green"
        elif hdi_high < rope_low: 
            recommendation = f"Review Solution {best_variant_idx+1} (Likely best but worse than Control within ROPE)"
            rec_style = "red"
        elif hdi_low >= rope_low and hdi_high <= rope_high: 
            if prob_s_beats_c > 0.99 and loss_ctrl_vs_sol > loss_sol_vs_ctrl * loss_ratio_threshold :
                 recommendation = f"Accept Solution {best_variant_idx+1} (Practically Equivalent to Control but High Confidence & Favorable Risk)"
                 rec_style = "green"
            else:
                recommendation = f"Solution {best_variant_idx+1} is Likely Best but Practically Equivalent to Control"
                rec_style = "blue"
        else: 
            recommendation = f"Accept Solution {best_variant_idx+1} (Strong Candidate)"
            rec_style = "yellow" 
            if prob_s_beats_c >= p_threshold and loss_ctrl_vs_sol > loss_sol_vs_ctrl * loss_ratio_threshold:
                recommendation = f"Accept Solution {best_variant_idx+1} (High P(>C) & Favorable Risk, despite ROPE overlap)"
                rec_style = "green"
        
        return evaluation, recommendation, rec_style

    def _get_dynamic_axis_range(self, *distributions_params_or_samples, 
                                percentile_low=0.01, percentile_high=99.99, 
                                padding_factor=0.08, allow_negative=False): 
        all_quantiles = np.array([])
        for item in distributions_params_or_samples:
            if item is None: continue 
            if isinstance(item, tuple) and len(item) == 2: 
                alpha, beta = item
                if alpha > 0 and beta > 0: 
                    q_low = stats.beta.ppf(percentile_low / 100.0, alpha, beta)
                    q_high = stats.beta.ppf(percentile_high / 100.0, alpha, beta)
                    if not np.isnan(q_low) and not np.isnan(q_high):
                         all_quantiles = np.concatenate([all_quantiles, [q_low, q_high]])
            elif isinstance(item, np.ndarray) and item.size > 0: 
                valid_samples = item[~np.isnan(item)]
                if valid_samples.size > 0:
                    q_low = np.percentile(valid_samples, percentile_low)
                    q_high = np.percentile(valid_samples, percentile_high)
                    all_quantiles = np.concatenate([all_quantiles, [q_low, q_high]])
        if all_quantiles.size == 0:
            return (0.0, 0.1) if not allow_negative else (-0.05, 0.05) 
        min_val = np.min(all_quantiles)
        max_val = np.max(all_quantiles)
        current_range = max_val - min_val
        if current_range < 1e-6: 
            padding = 0.005 
        else:
            padding = current_range * padding_factor 
        axis_min = min_val - padding
        if not allow_negative:
            axis_min = max(0.0, axis_min) 
        axis_max = max_val + padding
        if not allow_negative: 
             axis_max = min(1.0, axis_max)
        if axis_max <= axis_min: 
             axis_max = axis_min + (0.001 if not allow_negative else 0.0001 * abs(axis_min) + 0.0001) 
        if not allow_negative and axis_max > 1.0: axis_max = 1.0 
        if not allow_negative and axis_min < 0.0: axis_min = 0.0 
        return axis_min, axis_max


    def plot_distributions_plotly(self, rope_abs_diff=(-0.005, 0.005), rope_rel_lift=(-0.05, 0.05),
                                  n_samples_for_plot=10000):
        """
        Generate and display interactive plots for multiple variants.
        """
        samples_data = self.get_posterior_samples(n_samples=n_samples_for_plot)
        control_post_s = samples_data['control_rate']
        
        solution_line_colors = ['lightcoral', 'lightseagreen', 'mediumpurple', 'gold']
        solution_fill_colors = [
            'rgba(240,128,128,0.4)', 
            'rgba(32,178,170,0.4)',  
            'rgba(147,112,219,0.4)',
            'rgba(255,215,0,0.4)'   
        ]
        
        # Determine best solution for dynamic titles
        metrics_temp = self.calculate_metrics(rope_abs_diff, rope_rel_lift) 
        best_sol_idx = -1
        max_p_best = metrics_temp.get('prob_control_is_best', 0.0)
        best_sol_name_for_title = "Solution 1" # Default if no clear best or only one solution
        if self.num_solution_variants > 0:
            best_sol_name_for_title = metrics_temp['solutions'][0]['name'] # Default to first solution

        for i, sol_metrics in enumerate(metrics_temp['solutions']):
            if sol_metrics.get('prob_is_best', 0.0) > max_p_best:
                max_p_best = sol_metrics.get('prob_is_best', 0.0)
                best_sol_idx = i
                best_sol_name_for_title = sol_metrics['name']
        if best_sol_idx == -1 and self.num_solution_variants > 0: # If control was best, but we need a solution for title
            best_sol_name_for_title = metrics_temp['solutions'][0]['name'] # Fallback to first solution name for titles
        elif best_sol_idx == -1 and self.num_solution_variants == 0: # Should not happen with current logic
            best_sol_name_for_title = "Solution"


        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "<b>Prior Distributions</b>", 
                "<b>Observed Data Likelihoods</b>",
                "<b>Posterior Distributions</b>",
                f"<b>Which Variant is Most Likely the Winner?</b>", # Updated title
                f"<b>Absolute Difference: {best_sol_name_for_title} vs. Control</b>", # Updated title
                f"<b>Probability of {best_sol_name_for_title} Beating Control by > X% (Relative Lift)</b>" # Updated title
            ),
            specs=[[{}, {}], 
                   [{}, {}], 
                   [{}, {}]], 
            vertical_spacing=0.15, 
            horizontal_spacing=0.1
        )

        all_prior_params = [(self.control_prior_alpha, self.control_prior_beta)] + \
                           [(self.solution_prior_alpha[i], self.solution_prior_beta[i]) for i in range(self.num_solution_variants)]
        prior_min_x, prior_max_x = self._get_dynamic_axis_range(*all_prior_params, allow_negative=False)
        x_prior_plot = np.linspace(prior_min_x, prior_max_x, 200)

        all_like_params = [(self.control_observed_alpha_likelihood, self.control_observed_beta_likelihood) if self.control_samples > 0 else None] + \
                          [(self.solution_observed_alpha_likelihood[i], self.solution_observed_beta_likelihood[i]) if self.solution_samples[i] > 0 else None for i in range(self.num_solution_variants)]
        like_min_x, like_max_x = self._get_dynamic_axis_range(*[p for p in all_like_params if p is not None], allow_negative=False)
        x_like_plot = np.linspace(like_min_x, like_max_x, 200)

        all_post_samples = [control_post_s] + [samples_data[f"solution_{i+1}_rate"] for i in range(self.num_solution_variants)]
        post_min_x, post_max_x = self._get_dynamic_axis_range(*[s for s in all_post_samples if len(s) > 0], allow_negative=False)
        x_post_plot = np.linspace(post_min_x, post_max_x, 200)

        # Plot 1: Prior Distributions
        fig.add_trace(go.Scatter(x=x_prior_plot, y=stats.beta.pdf(x_prior_plot, self.control_prior_alpha, self.control_prior_beta),
                                 mode='lines', name='Ctrl Prior', legendgroup="Priors", line=dict(dash='dash', color='skyblue', width=2),
                                 hovertemplate="<b>Ctrl Prior</b><br>Rate: %{x:.3%}<br>Density: %{y:.2f}<extra></extra>"), row=1, col=1)
        for i in range(self.num_solution_variants):
            fig.add_trace(go.Scatter(x=x_prior_plot, y=stats.beta.pdf(x_prior_plot, self.solution_prior_alpha[i], self.solution_prior_beta[i]),
                                     mode='lines', name=f'Sol {i+1} Prior', legendgroup="Priors", 
                                     line=dict(dash='dash', color=solution_line_colors[i % len(solution_line_colors)], width=2),
                                     hovertemplate=f"<b>Sol {i+1} Prior</b><br>Rate: %{{x:.3%}}<br>Density: %{{y:.2f}}<extra></extra>"), row=1, col=1)
        fig.update_xaxes(range=[prior_min_x, prior_max_x], row=1, col=1) 

        # Plot 2: Observed Data Likelihoods
        if self.control_samples > 0:
            fig.add_trace(go.Scatter(x=x_like_plot, y=stats.beta.pdf(x_like_plot, self.control_observed_alpha_likelihood, self.control_observed_beta_likelihood),
                                     mode='lines', name='Ctrl Likelihood', legendgroup="Likelihoods", line=dict(dash='dot', color='lightgreen', width=2),
                                     hovertemplate="<b>Ctrl Likelihood</b><br>Rate: %{x:.3%}<br>Density: %{y:.2f}<extra></extra>"), row=1, col=2)
        for i in range(self.num_solution_variants):
            if self.solution_samples[i] > 0:
                fig.add_trace(go.Scatter(x=x_like_plot, y=stats.beta.pdf(x_like_plot, self.solution_observed_alpha_likelihood[i], self.solution_observed_beta_likelihood[i]),
                                         mode='lines', name=f'Sol {i+1} Likelihood', legendgroup="Likelihoods", 
                                         line=dict(dash='dot', color=solution_line_colors[i % len(solution_line_colors)], width=1.5),
                                         opacity=0.8, 
                                         hovertemplate=f"<b>Sol {i+1} Likelihood</b><br>Rate: %{{x:.3%}}<br>Density: %{{y:.2f}}<extra></extra>"), row=1, col=2)
        if self.control_samples == 0 and all(s == 0 for s in self.solution_samples):
             fig.add_annotation(text="No observed data entered", showarrow=False, row=1, col=2)
        fig.update_xaxes(range=[like_min_x, like_max_x], row=1, col=2) 

        # Plot 3: Posterior Distributions
        max_density_post = 0
        if len(control_post_s) > 1: 
            kde_control = stats.gaussian_kde(control_post_s)
            y_kde_control = kde_control(x_post_plot)
            max_density_post = max(max_density_post, np.max(y_kde_control))
            fig.add_trace(go.Scatter(x=x_post_plot, y=y_kde_control, mode='lines', name='Ctrl Posterior', legendgroup="Posteriors", fill='tozeroy',
                                     fillcolor='rgba(70,130,180,0.4)', line=dict(color='steelblue', width=2),
                                     hovertemplate="<b>Ctrl Posterior</b><br>Rate: %{x:.3%}<br>Density: %{y:.2f}<extra></extra>"), row=2, col=1)
        for i in range(self.num_solution_variants):
            sol_s = samples_data[f"solution_{i+1}_rate"]
            if len(sol_s) > 1:
                kde_solution = stats.gaussian_kde(sol_s)
                y_kde_solution = kde_solution(x_post_plot)
                max_density_post = max(max_density_post, np.max(y_kde_solution))
                fig.add_trace(go.Scatter(x=x_post_plot, y=y_kde_solution, mode='lines', name=f'Sol {i+1} Posterior', legendgroup="Posteriors", fill='tozeroy',
                                         fillcolor=solution_fill_colors[i % len(solution_fill_colors)], 
                                         line=dict(color=solution_line_colors[i % len(solution_line_colors)], width=2),
                                         hovertemplate=f"<b>Sol {i+1} Posterior</b><br>Rate: %{{x:.3%}}<br>Density: %{{y:.2f}}<extra></extra>"), row=2, col=1)
        fig.update_xaxes(range=[post_min_x, post_max_x], row=2, col=1) 

        # Plot 4: Probability of Being Best
        prob_best_names = [self.variant_names[0]] + [sol_metrics['name'] for sol_metrics in metrics_temp['solutions']]
        prob_best_values = [metrics_temp.get('prob_control_is_best', 0)] + [sol_metrics.get('prob_is_best', 0) for sol_metrics in metrics_temp['solutions']]
        bar_colors = ['skyblue'] + [solution_line_colors[i % len(solution_line_colors)] for i in range(self.num_solution_variants)] 
        fig.add_trace(go.Bar(x=prob_best_names, y=prob_best_values, name='P(Best)', legendgroup="P(Best)",
                             marker_color=bar_colors, text=[f"{p:.1%}" for p in prob_best_values], textposition='auto',
                             hovertemplate="<b>%{x}</b><br>P(Best): %{y:.2%}<extra></extra>"), row=2, col=2)
        fig.update_yaxes(tickformat=".0%", range=[0,1.05], row=2, col=2)

        # Plot 5: Difference (Best Solution vs Control)
        best_sol_abs_diff_s = np.array([])
        # Title already set dynamically via subplot_titles and best_sol_name_for_title
        if best_sol_idx != -1: 
            best_sol_abs_diff_s = samples_data[f"abs_diff_s{best_sol_idx+1}_c"]
        elif self.num_solution_variants > 0 : 
            best_sol_abs_diff_s = samples_data.get(f"abs_diff_s1_c", np.array([]))
        
        if len(best_sol_abs_diff_s) > 1:
            diff_min_x, diff_max_x = self._get_dynamic_axis_range(best_sol_abs_diff_s, allow_negative=True) 
            x_diff_plot = np.linspace(diff_min_x, diff_max_x, 200)
            kde_abs_diff = stats.gaussian_kde(best_sol_abs_diff_s)
            y_kde_abs_diff = kde_abs_diff(x_diff_plot)
            fig.add_trace(go.Scatter(x=x_diff_plot, y=y_kde_abs_diff, mode='lines', name='Abs. Diff (Best Sol)', legendgroup="Difference Analysis", fill='tozeroy',
                                     fillcolor='rgba(128,0,128,0.4)', line=dict(color='purple', width=2),
                                     hovertemplate="<b>Abs. Difference</b><br>Value: %{x:.3%}<br>Density: %{y:.2f}<extra></extra>"), row=3, col=1) 
            abs_diff_mean = np.mean(best_sol_abs_diff_s)
            abs_diff_hdi = _calculate_hdi(best_sol_abs_diff_s)
            fig.add_vline(x=abs_diff_mean, line_width=1.5, line_dash="dash", line_color="indigo", row=3, col=1)
            fig.add_vline(x=abs_diff_hdi[0], line_width=1.5, line_dash="dot", line_color="indigo", row=3, col=1)
            fig.add_vline(x=abs_diff_hdi[1], line_width=1.5, line_dash="dot", line_color="indigo", row=3, col=1) 
            fig.add_shape(type="rect", x0=rope_abs_diff[0], x1=rope_abs_diff[1], y0=0, y1=np.max(y_kde_abs_diff)*1.1 if len(y_kde_abs_diff) > 0 else 1,
                          fillcolor="rgba(169,169,169,0.3)", opacity=0.3, layer="below", line_width=0, name="ROPE Abs.Diff.", row=3, col=1)
            fig.update_xaxes(range=[diff_min_x, diff_max_x], row=3, col=1) 

        # Plot 6: Cumulative P(Best Solution Relative Uplift > X)
        best_sol_rel_lift_s = np.array([])
        # Title already set dynamically
        if best_sol_idx != -1:
            best_sol_rel_lift_s = samples_data[f"rel_lift_s{best_sol_idx+1}_c"][~np.isnan(samples_data[f"rel_lift_s{best_sol_idx+1}_c"])]
        elif self.num_solution_variants > 0:
            best_sol_rel_lift_s = samples_data.get(f"rel_lift_s1_c", np.array([]))
            best_sol_rel_lift_s = best_sol_rel_lift_s[~np.isnan(best_sol_rel_lift_s)]

        if len(best_sol_rel_lift_s) > 0:
            cum_rel_min_x, cum_rel_max_x = self._get_dynamic_axis_range(best_sol_rel_lift_s, allow_negative=True)
            sorted_rel_lift = np.sort(best_sol_rel_lift_s)
            y_cumulative_rel = 1. - (np.arange(len(sorted_rel_lift)) / float(len(sorted_rel_lift)))
            fig.add_trace(go.Scatter(x=sorted_rel_lift, y=y_cumulative_rel, mode='lines', name='P(Rel. Uplift > X)', legendgroup="Cumulative Uplift", line=dict(color='darkcyan', width=2),
                                     hovertemplate="<b>P(Rel. Uplift > X)</b><br>Rel. Uplift (X): %{x:.2%}<br>Probability: %{y:.2%}<extra></extra>"), row=3, col=2) 
            fig.add_hline(y=0.95, line_width=1, line_dash="dash", line_color="gray", row=3, col=2)
            fig.add_hline(y=0.50, line_width=1, line_dash="dot", line_color="gray", row=3, col=2)
            fig.add_vline(x=0, line_width=1, line_dash="solid", line_color="black", row=3, col=2)
            fig.update_xaxes(range=[cum_rel_min_x, cum_rel_max_x], row=3, col=2)
            fig.update_yaxes(range=[0,1.05], row=3, col=2)
        else:
             fig.add_annotation(text="Not enough data for Cumulative Rel. Uplift", showarrow=False, row=3, col=2)

        fig.update_layout(
            height=1200, 
            title_text="<b>Bayesian A/B Test Visualizations</b>", title_x=0.5, title_font_size=20,
            legend_traceorder='grouped', legend_tracegroupgap=15, hovermode='x unified', template='plotly_white'
        )
        for r,c in [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2)]: fig.update_xaxes(tickformat=".2%", row=r, col=c)
        for r,c in [(1,1),(1,2),(2,1),(3,1),(3,2)]: fig.update_yaxes(title_text="Density", row=r, col=c)
        fig.update_yaxes(title_text="Probability P(Best)", tickformat=".0%", row=2, col=2)
        fig.update_yaxes(title_text="Probability P(Rel. Uplift > X)", tickformat=".0%", row=3, col=2) 
        
        fig.show()

# --- Input Helper Functions ---
# ... (No changes to input functions)
def get_float_input(console, prompt, default_value=None, is_positive=False, is_percentage_for_rope=False):
    """Helper function to get a validated float input from the user."""
    while True:
        try:
            input_prompt = prompt
            if is_percentage_for_rope and "percentage" not in prompt.lower() and "(e.g., 5 for 5%)" not in prompt:
                input_prompt = prompt + " (e.g., 5 for 5%)"
            
            val_str = console.input(f"[b]{input_prompt}[/b]" + (f" (default: {default_value})" if default_value is not None else "") + ": ")
            
            if not val_str and default_value is not None:
                return float(default_value)

            val = float(val_str)

            if is_percentage_for_rope: 
                val = val / 100.0 
            
            if is_positive and val <= 0:
                if not (is_percentage_for_rope and "lower bound" in prompt.lower()):
                     console.print("Value must be positive.", style="bold red")
                     continue
            
            if is_percentage_for_rope and (val < -1.0 or val > 1.0): 
                 console.print("Percentage for ROPE seems too large/small. Value should be decimal (e.g. 0.05 for 5%).", style="yellow")
            return val
        except ValueError:
            console.print("Invalid input. Please enter a number.", style="bold red")


def get_int_input(console, prompt, default_value=None, min_val=0):
    while True:
        try:
            val_str = console.input(f"[b]{prompt}[/b]" + (f" (default: {default_value})" if default_value is not None else "") + ": ")
            if not val_str and default_value is not None:
                return int(default_value)
            val = int(val_str)
            if val < min_val:
                console.print(f"Value must be {min_val} or greater.", style="bold red")
                continue
            return val
        except ValueError:
            console.print("Invalid input. Please enter a whole number.", style="bold red")

def get_user_inputs(console):
    """Gets all necessary inputs from the user with validation."""
    console.print(Panel(Text("Bayesian A/B Test Analyzer", justify="center", style="bold white on blue"), expand=False))
    
    num_solution_variants = get_int_input(console, "How many solution variants are you testing (in addition to Control)?", default_value=1, min_val=1)

    console.print("\nPlease enter your prior beliefs and test data:\n")
    console.print(Panel("[bold underline]Priors (Beta Distribution Parameters)[/bold underline]\n"
                        "Priors represent your beliefs before seeing the test data.\n"
                        "Alpha (α) can be thought of as prior 'successes' + 1 (or a strength parameter).\n"
                        "Beta (β) can be thought of as prior 'failures' + 1 (or a strength parameter).\n"
                        "The ratio α / (α + β) is the prior mean conversion rate.\n"
                        "Larger α and β mean a stronger (more confident) prior.\n"
                        "Using alpha=1, beta=1 is a common 'uninformative' prior (uniform distribution).", 
                        title="Prior Information", border_style="cyan", expand=False))
    
    control_prior_alpha = get_float_input(console, "Control Group Prior Alpha", default_value=1.0, is_positive=True)
    control_prior_beta = get_float_input(console, "Control Group Prior Beta", default_value=1.0, is_positive=True)

    solution_prior_alphas = []
    solution_prior_betas = []

    auto_sol_prior_all = console.input(f"Auto-generate priors for all {num_solution_variants} Solution variants based on Control priors? (y/n, default: y): ").lower().strip()
    
    for i in range(num_solution_variants):
        console.print(f"\n--- Solution Variant {i+1} Priors ---")
        if auto_sol_prior_all == "" or auto_sol_prior_all == "y":
            solution_prior_total_n = get_int_input(console, f"Total pseudo-observations for Solution {i+1} prior (default: 20)", default_value=20, min_val=2)
            if control_prior_alpha + control_prior_beta == 0: 
                console.print("Control priors sum to zero, cannot derive solution priors. Please enter manually.", style="yellow")
                s_alpha = get_float_input(console, f"Solution {i+1} Prior Alpha", default_value=1.0, is_positive=True)
                s_beta = get_float_input(console, f"Solution {i+1} Prior Beta", default_value=1.0, is_positive=True)
            else:
                control_prior_rate = control_prior_alpha / (control_prior_alpha + control_prior_beta)
                s_alpha_candidate = control_prior_rate * solution_prior_total_n
                s_beta_candidate = (1 - control_prior_rate) * solution_prior_total_n
                if s_alpha_candidate < 1.0: s_alpha = 1.0; s_beta = float(max(1.0, solution_prior_total_n - 1.0)) 
                elif s_beta_candidate < 1.0: s_beta = 1.0; s_alpha = float(max(1.0, solution_prior_total_n - 1.0)) 
                else: s_alpha = s_alpha_candidate; s_beta = s_beta_candidate
                if s_alpha < 1.0: s_alpha = 1.0
                if s_beta < 1.0: s_beta = 1.0
                current_sum = s_alpha + s_beta
                if not np.isclose(current_sum, solution_prior_total_n): 
                    if current_sum < solution_prior_total_n:
                        if s_alpha > s_beta: s_alpha += (solution_prior_total_n - current_sum)
                        else: s_beta += (solution_prior_total_n - current_sum)
                console.print(f"Derived Solution {i+1} Priors: Alpha={s_alpha:.2f}, Beta={s_beta:.2f}", style="dim")
        else:
            s_alpha = get_float_input(console, f"Solution {i+1} Prior Alpha", default_value=1.0, is_positive=True)
            s_beta = get_float_input(console, f"Solution {i+1} Prior Beta", default_value=1.0, is_positive=True)
        solution_prior_alphas.append(s_alpha)
        solution_prior_betas.append(s_beta)

    console.print(Panel("[bold underline]Test Results[/bold underline]", title="Observed Data", border_style="cyan", expand=False))
    current_control_samples = get_int_input(console, "Control Group Samples")
    current_control_conversions = get_int_input(console, "Control Group Conversions", min_val=0)
    while current_control_conversions > current_control_samples:
        console.print("Conversions cannot exceed samples. Please re-enter for Control Group.", style="bold red")
        current_control_conversions = get_int_input(console, "Control Group Conversions", min_val=0)

    current_solution_samples_list = []
    current_solution_conversions_list = []
    for i in range(num_solution_variants):
        console.print(f"\n--- Solution Variant {i+1} Test Results ---")
        s_samples = get_int_input(console, f"Solution {i+1} Samples")
        s_conversions = get_int_input(console, f"Solution {i+1} Conversions", min_val=0)
        while s_conversions > s_samples:
            console.print(f"Conversions cannot exceed samples for Solution {i+1}.", style="bold red")
            s_conversions = get_int_input(console, f"Solution {i+1} Conversions", min_val=0)
        current_solution_samples_list.append(s_samples)
        current_solution_conversions_list.append(s_conversions)

    console.print(Panel("[bold underline]ROPE (Region of Practical Equivalence)[/bold underline]\n"
                        "Define a range where differences are considered practically insignificant. "
                        "The absolute difference ROPE will be derived from the relative lift ROPE.",
                        title="ROPE Definition", border_style="cyan", expand=False))
    rope_rel_lift_upper_pct_val = get_float_input(console, "ROPE Relative Lift - Symmetrical Boundary (% of Control Rate, e.g., 5 for 5%)", 
                                                  default_value=2.0, # Changed default to 2.0
                                                  is_positive=True, is_percentage_for_rope=True)
    rope_rel_lift_upper = rope_rel_lift_upper_pct_val 
    rope_rel_lift_lower = -rope_rel_lift_upper
    rope_rel_lift = (rope_rel_lift_lower, rope_rel_lift_upper)
    console.print(f"Relative Lift ROPE set to: ({rope_rel_lift_lower:.2%}, {rope_rel_lift_upper:.2%})", style="dim")

    rope_abs_diff_lower = None; rope_abs_diff_high = None
    if current_control_samples > 0 and current_control_samples >= current_control_conversions :
        control_observed_rate = current_control_conversions / current_control_samples
        if control_observed_rate > 1e-9: 
            abs_delta = rope_rel_lift_upper * control_observed_rate
            rope_abs_diff_lower = -abs_delta; rope_abs_diff_high = abs_delta
            console.print(f"Derived Absolute Difference ROPE: ({rope_abs_diff_lower:.3%}, {rope_abs_diff_high:.3%}) (based on Control Rate of {control_observed_rate:.2%})", style="dim")
        else: console.print(f"Control observed rate is effectively 0. Cannot derive absolute ROPE from relative ROPE.", style="yellow")
    else: console.print(f"Control samples are 0. Cannot derive absolute ROPE from relative ROPE.", style="yellow")
    if rope_abs_diff_lower is None or rope_abs_diff_high is None: 
        console.print(f"Using default Absolute Difference ROPE because it could not be derived.", style="yellow")
        default_abs_rope_delta = 0.001 
        rope_abs_diff_lower = -default_abs_rope_delta; rope_abs_diff_high = default_abs_rope_delta
        console.print(f"Default Absolute Difference ROPE set to: ({rope_abs_diff_lower:.3%}, {rope_abs_diff_high:.3%})", style="dim")
    rope_abs_diff = (rope_abs_diff_lower, rope_abs_diff_high)

    return {
        "num_solution_variants": num_solution_variants,
        "control_prior_alpha": control_prior_alpha, "control_prior_beta": control_prior_beta,
        "solution_prior_alphas": solution_prior_alphas, "solution_prior_betas": solution_prior_betas,
        "control_samples": current_control_samples, "control_conversions": current_control_conversions,
        "solution_samples_list": current_solution_samples_list, "solution_conversions_list": current_solution_conversions_list,
        "rope_abs_diff": rope_abs_diff, "rope_rel_lift": rope_rel_lift
    }

# --- Display Helper Functions ---
def display_test_outcomes_table(console, metrics):
    """Displays the Test Outcomes table for multiple variants."""
    table = Table(title="Test Outcomes Summary", title_style="bold magenta", border_style="blue")
    table.add_column("Group", style="cyan")
    table.add_column("Win Rate (Mean)", style="dim")
    table.add_column("Rel. Lift vs Ctrl (Mean)", style="dim") 
    table.add_column("95% HDI (Rate)", style="dim")
    
    # Control Row
    c_metrics = metrics['control']
    table.add_row("Control", f"{c_metrics['posterior_mean_rate']:.2%}", "N/A", f"[{c_metrics['rate_hdi'][0]:.2%}, {c_metrics['rate_hdi'][1]:.2%}]")
    
    # Solution Variant Rows
    for sol_metrics in metrics['solutions']:
        rl_mean = sol_metrics.get('relative_lift_mean', np.nan)
        table.add_row(
            sol_metrics['name'],
            f"{sol_metrics['posterior_mean_rate']:.2%}",
            f"{rl_mean:+.2%}" if not np.isnan(rl_mean) else "N/A", 
            f"[{sol_metrics['rate_hdi'][0]:.2%}, {sol_metrics['rate_hdi'][1]:.2%}]"
        )
    console.print(Padding(table, (1, 0)))


def display_confidence_intervals_summary(console, metrics):
    """Displays a dedicated summary of key confidence intervals for multiple variants."""
    panel_content = Text()
    # Control
    c_metrics = metrics['control']
    panel_content.append("Control Conversion Rate:\n", style="bold sky_blue3")
    panel_content.append(f"  Mean: {c_metrics['posterior_mean_rate']:.2%}, 95% HDI: [{c_metrics['rate_hdi'][0]:.2%}, {c_metrics['rate_hdi'][1]:.2%}]\n\n")

    # Solution Variants
    for sol_metrics in metrics['solutions']:
        panel_content.append(f"{sol_metrics['name']} Conversion Rate:\n", style="bold light_coral")
        panel_content.append(f"  Mean: {sol_metrics['posterior_mean_rate']:.2%}, 95% HDI: [{sol_metrics['rate_hdi'][0]:.2%}, {sol_metrics['rate_hdi'][1]:.2%}]\n\n")
        
        panel_content.append(f"Abs. Diff ({sol_metrics['name']} - Control):\n", style="bold dark_violet")
        panel_content.append(f"  Mean: {sol_metrics['absolute_difference_mean']:.2%}, 95% HDI: [{sol_metrics['absolute_difference_hdi'][0]:.2%}, {sol_metrics['absolute_difference_hdi'][1]:.2%}]\n")
        
        rl_mean_val = sol_metrics.get('relative_lift_mean', np.nan)
        rl_hdi_low_val, rl_hdi_high_val = sol_metrics.get('relative_lift_hdi', (np.nan, np.nan))
        if not np.isnan(rl_mean_val):
            panel_content.append(f"Rel. Lift (({sol_metrics['name']}-Ctrl)/Ctrl):\n", style="bold green4")
            panel_content.append(f"  Mean: {rl_mean_val:.2%}, 95% HDI: [{rl_hdi_low_val:.2%}, {rl_hdi_high_val:.2%}]\n\n")
        else:
            panel_content.append("\n")


    console.print(Panel(panel_content, title="[bold]Confidence Intervals (95% HDI)[/bold]", border_style="steel_blue", expand=False)) 


def display_detailed_metrics(console, metrics, rope_abs_diff, rope_rel_lift):
    panel_content = Text()
    
    panel_content.append("Probability of Being Best Overall:\n", style="bold underline")
    panel_content.append(f"  Control: {metrics.get('prob_control_is_best', 0.0):.2%}\n")
    for sol_metrics in metrics['solutions']:
        panel_content.append(f"  {sol_metrics['name']}: {sol_metrics.get('prob_is_best', 0.0):.2%}\n")
    panel_content.append("\n")

    for i, sol_metrics in enumerate(metrics['solutions']):
        panel_content.append(f"--- Analysis for {sol_metrics['name']} vs Control ---\n", style="bold yellow")
        panel_content.append("Probabilities:\n", style="bold underline")
        panel_content.append(f"  P({sol_metrics['name']} > Control): {sol_metrics['prob_beats_control']:.2%}\n")
        panel_content.append(f"  P({sol_metrics['name']} > Ctrl + {metrics.get('prob_beat_threshold_value',0.0):.1%}): {sol_metrics['prob_beats_control_by_threshold']:.2%}\n\n")
        
        panel_content.append(f"ROPE Analysis (Absolute Difference: {rope_abs_diff[0]:.2%} to {rope_abs_diff[1]:.2%}):\n", style="bold underline")
        panel_content.append(f"  P(Diff < ROPE Low): {sol_metrics['prob_abs_diff_below_rope']:.2%}\n")
        panel_content.append(f"  P(Diff In ROPE):   {sol_metrics['prob_abs_diff_in_rope']:.2%}\n")
        panel_content.append(f"  P(Diff > ROPE High):{sol_metrics['prob_abs_diff_above_rope']:.2%}\n\n")

        if not np.isnan(sol_metrics.get('prob_rel_lift_in_rope', np.nan)): 
            panel_content.append(f"ROPE Analysis (Relative Lift: {rope_rel_lift[0]:.1%} to {rope_rel_lift[1]:.1%}):\n", style="bold underline")
            panel_content.append(f"  P(Lift < ROPE Low): {sol_metrics.get('prob_rel_lift_below_rope', np.nan):.2%}\n")
            panel_content.append(f"  P(Lift In ROPE):   {sol_metrics.get('prob_rel_lift_in_rope', np.nan):.2%}\n")
            panel_content.append(f"  P(Lift > ROPE High):{sol_metrics.get('prob_rel_lift_above_rope', np.nan):.2%}\n\n")

        panel_content.append(f"Expected Loss ({sol_metrics['name']} vs Control):\n", style="bold underline")
        panel_content.append(f"  Choosing {sol_metrics['name']} (if Control is better): {sol_metrics['expected_loss_vs_control_choosing_solution']:.4%}\n")
        panel_content.append(f"  Choosing Control (if {sol_metrics['name']} is better): {sol_metrics['expected_loss_vs_control_choosing_control']:.4%}\n\n")
    
    console.print(Panel(panel_content, title="[bold]Further Analysis Details[/bold]", border_style="green", expand=False)) 

def display_explanations(console):
    text = Text()
    text.append("Key Concepts:\n\n", style="bold underline")
    text.append("ROPE (Region of Practical Equivalence):\n", style="bold cyan")
    text.append("  The range of differences you consider too small to matter. If the credible interval for the difference falls mostly within ROPE, the variants are practically equivalent.\n\n")
    text.append("HDI (Highest Density Interval):\n", style="bold cyan")
    text.append("  The range containing a specific percentage (e.g., 95%) of the most credible values for a parameter (e.g., conversion rate or difference). We can say there's a 95% probability the true value lies within the 95% HDI.\n\n")
    
    text.append("Interpreting 'Further Analysis Details':\n", style="bold underline")
    text.append("  - Probability of Being Best: For each variant, the chance it has the highest true conversion rate among all tested variants (including Control).\n")
    text.append("  - Probabilities (vs Control): Shows the likelihood of a solution variant being better than Control, or better by a certain threshold.\n")
    text.append("  - ROPE Analysis (vs Control): Shows the probability that the true difference/lift between a solution and Control falls below, within, or above your defined ROPE.\n")
    text.append("  - Expected Loss (vs Control): Estimates the average 'cost' of making the wrong decision between a specific solution and Control.\n\n")

    text.append("Interpreting Charts:\n", style="bold underline")
    text.append("  - Prior plots show initial beliefs for Control and all Solutions, overlaid.\n")
    text.append("  - Likelihood plots show what current test data suggests for each variant, overlaid.\n")
    text.append("  - Posterior plots combine priors and likelihood for updated beliefs, overlaid. HDIs are marked as small shaded regions at the base.\n")
    text.append("  - P(Best) Bar Chart: Visualizes the probability of each variant being the overall best.\n")
    text.append("  - Difference plots show distributions of (Solution - Control) for the best solution or a selected one.\n")
    text.append("  - Cumulative P(Uplift > X) plot shows the probability that the true relative uplift (for the best solution vs Control) is greater than X.\n")
    text.append("  - Hover over chart elements for specific values.\n")
    console.print(Panel(text, title="[bold]Understanding the Results[/bold]", border_style="magenta", expand=False))


if __name__ == "__main__":
    console = Console()
    try:
        inputs = get_user_inputs(console) 
        
        experiment = BayesianExperiment(num_solution_variants=inputs["num_solution_variants"])
        experiment.set_priors(
            inputs["control_prior_alpha"], inputs["control_prior_beta"],
            inputs["solution_prior_alphas"], inputs["solution_prior_betas"]
        )
        experiment.update_results(
            inputs["control_samples"], inputs["control_conversions"],
            inputs["solution_samples_list"], inputs["solution_conversions_list"]
        )

        console.print("\n[bold]Calculating metrics...[/bold]\n")
        prob_beat_thresh_val = 0.001 
        metrics = experiment.calculate_metrics(
            rope_abs_diff=inputs["rope_abs_diff"],
            rope_rel_lift=inputs["rope_rel_lift"],
            prob_beat_threshold=prob_beat_thresh_val 
        )
        metrics['prob_beat_threshold_value'] = prob_beat_thresh_val 
        
        evaluation, recommendation, rec_style = experiment.get_decision_summary(
            metrics, 
            inputs["rope_abs_diff"], # ROPE for Solution vs Control
            p_threshold=0.95, 
            loss_ratio_threshold=5
            )
        summary_panel_text = Text()
        summary_panel_text.append("Evaluation: ", style="bold")
        summary_panel_text.append(f"{evaluation}\n", style=f"bold {rec_style}")
        summary_panel_text.append("Recommendation: ", style="bold")
        summary_panel_text.append(f"{recommendation}", style=f"bold {rec_style}")
        console.print(Panel(summary_panel_text, title="[bold blue]Decision Summary[/bold blue]", expand=False, border_style=rec_style))

        display_test_outcomes_table(console, metrics)
        display_confidence_intervals_summary(console, metrics) # Will show for control and all solutions
        display_detailed_metrics(console, metrics, inputs["rope_abs_diff"], inputs["rope_rel_lift"])
        
        console.print(Panel(Text("Visualizations", justify="center"), title="[bold]Charts[/bold]", border_style="yellow", expand=False))
        experiment.plot_distributions_plotly( 
            rope_abs_diff=inputs["rope_abs_diff"],
            rope_rel_lift=inputs["rope_rel_lift"] 
        )
        
        display_explanations(console)
        console.print("\n[bold green]Analysis Complete.[/bold green]")

    except ValueError as ve:
        console.print(f"[bold red]Input Error:[/bold red] {ve}")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")

