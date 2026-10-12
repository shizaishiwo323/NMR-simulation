function sample16_all_nmr_brd_alpha1(inventory_csv, experiment_decay, output_dir, brd_dir)
% Reinvert the eight complete grouped decays and one experiment with MATLAB BRD.
runs = readtable(inventory_csv);
assert(height(runs) == 8, 'Expected eight grouped simulation runs');
for i = 1:height(runs)
    run_dir = fullfile(output_dir, sprintf('simulation_%02d', i));
    sample16_brd_alpha_sweep(string(runs.source_decay(i)), run_dir, brd_dir, 1, true);
    fprintf('simulation %d/8 rho2=%g um/s\n', i, runs.rho2_um_per_s(i));
end
sample16_brd_alpha_sweep(experiment_decay, fullfile(output_dir, 'experiment'), brd_dir, 1, true);
end
