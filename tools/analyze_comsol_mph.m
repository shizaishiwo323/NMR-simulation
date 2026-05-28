function analyze_comsol_mph(model_path, out_path, port)
%ANALYZE_COMSOL_MPH Read-only summary of a COMSOL MPH model via LiveLink.

if nargin < 3
    port = 20463;
end

addpath('C:\Program Files\COMSOL\COMSOL63\Multiphysics\mli');
diary(out_path);
cleanup_diary = onCleanup(@() diary('off'));

fprintf('COMSOL MPH read-only summary\n');
fprintf('Model path: %s\n', model_path);
fprintf('Output path: %s\n', out_path);
fprintf('MATLAB version: %s\n', version);
fprintf('Started: %s\n\n', datestr(now, 31));

try
    mphstart('localhost', port);
    fprintf('Connected to COMSOL mphserver on port %d.\n', port);
catch err
    fprintf('ERROR: Could not connect to COMSOL mphserver on port %d.\n%s\n', port, getReport(err, 'extended', 'hyperlinks', 'off'));
    return
end

try
    model = mphload(model_path, 'Model');
    fprintf('Loaded model successfully.\n\n');
catch err
    fprintf('ERROR: Could not load model.\n%s\n', getReport(err, 'extended', 'hyperlinks', 'off'));
    return
end

print_model_meta(model);
print_tag_collection('Parameters', @() model.param().tags);
print_component_tree(model);
print_tag_collection('Selections', @() model.selection().tags);
print_tag_collection('Functions', @() model.func().tags);
print_tag_collection('Materials', @() model.material().tags);
print_tag_collection('Studies', @() model.study().tags);
print_tag_collection('Solutions', @() model.sol().tags);
print_tag_collection('Datasets', @() model.result().dataset().tags);
print_tag_collection('Plot groups', @() model.result().tags);
print_tag_collection('Tables', @() model.result().table().tags);
print_constant_values(model);
print_function_details(model);
print_study_details(model);
print_solution_nodes(model);
print_result_details(model);
print_mesh_stats(model);
print_solution_info(model);

fprintf('\nFinished: %s\n', datestr(now, 31));
end

function print_model_meta(model)
fprintf('== Model metadata ==\n');
try_print('Model tag', @() char(model.tag));
try_print('Model label', @() char(model.label));
try_print('Model path', @() char(model.modelPath));
try
    fprintf('COMSOL version: %s\n', mphversion);
catch
end
fprintf('\n');
end

function print_component_tree(model)
fprintf('== Components ==\n');
comp_tags = get_tags(@() model.component().tags);
print_tags(comp_tags);
for i = 1:numel(comp_tags)
    comp = model.component(comp_tags{i});
    fprintf('\n-- Component %s --\n', comp_tags{i});
    try_print('Label', @() char(comp.label));
    print_tag_collection('Geometries', @() comp.geom().tags);
    print_tag_collection('Meshes', @() comp.mesh().tags);
    print_tag_collection('Physics', @() comp.physics().tags);
    print_tag_collection('Component variables', @() comp.variable().tags);
    print_tag_collection('Component selections', @() comp.selection().tags);
    try
        geom_tags = cellstr(comp.geom().tags);
        for g = 1:numel(geom_tags)
            geom = comp.geom(geom_tags{g});
            fprintf('Geometry %s: ', geom_tags{g});
            try
                fprintf('dimension=%s, ', char(geom.getSDim));
            catch
            end
            try_print_inline('label', @() char(geom.label));
            fprintf('\n');
        end
    catch
    end
    try
        phys_tags = cellstr(comp.physics().tags);
        for p = 1:numel(phys_tags)
            phys = comp.physics(phys_tags{p});
            fprintf('Physics %s: ', phys_tags{p});
            try_print_inline('type', @() char(phys.getType));
            try_print_inline('label', @() char(phys.label));
            fprintf('\n');
            print_feature_details(phys, sprintf('Physics %s features', phys_tags{p}));
        end
    catch
    end
end
fprintf('\n');
end

function print_constant_values(model)
fprintf('== Evaluated constants used by this workflow ==\n');
names = {'D', 'T1', 'T2', 'm0', 'tao', 'tao2'};
for i = 1:numel(names)
    try
        value = mphevaluate(model, names{i});
        fprintf('%s = ', names{i});
        disp(value);
    catch err
        fprintf('%s unavailable: %s\n', names{i}, err.message);
    end
end
fprintf('\n');
end

function print_function_details(model)
fprintf('== Function details ==\n');
tags = get_tags(@() model.func().tags);
for i = 1:numel(tags)
    node = model.func(tags{i});
    fprintf('-- Function %s --\n', tags{i});
    try_print('Label', @() char(node.label));
    try_print('Type', @() char(node.getType));
    try_disp_properties(node);
end
fprintf('\n');
end

function print_study_details(model)
fprintf('== Study details ==\n');
tags = get_tags(@() model.study().tags);
for i = 1:numel(tags)
    node = model.study(tags{i});
    fprintf('-- Study %s --\n', tags{i});
    try_print('Label', @() char(node.label));
    print_feature_details(node, sprintf('Study %s features', tags{i}));
end
fprintf('\n');
end

function print_solution_nodes(model)
fprintf('== Solution node details ==\n');
tags = get_tags(@() model.sol().tags);
for i = 1:numel(tags)
    node = model.sol(tags{i});
    fprintf('-- Solution %s --\n', tags{i});
    try_print('Label', @() char(node.label));
    print_feature_details(node, sprintf('Solution %s features', tags{i}));
end
fprintf('\n');
end

function print_result_details(model)
fprintf('== Result details ==\n');
plot_tags = get_tags(@() model.result().tags);
for i = 1:numel(plot_tags)
    node = model.result(plot_tags{i});
    fprintf('-- Plot group %s --\n', plot_tags{i});
    try_print('Label', @() char(node.label));
    try_print('Type', @() char(node.getType));
    try_disp_properties(node);
    print_feature_details(node, sprintf('Plot group %s features', plot_tags{i}));
end
table_tags = get_tags(@() model.result().table().tags);
for i = 1:numel(table_tags)
    node = model.result().table(table_tags{i});
    fprintf('-- Table %s --\n', table_tags{i});
    try_print('Label', @() char(node.label));
    try_disp_properties(node);
end
fprintf('\n');
end

function print_feature_details(node, title_text)
fprintf('== %s ==\n', title_text);
tags = get_tags(@() node.feature().tags);
if isempty(tags)
    fprintf('(none or unavailable)\n');
    return
end
for i = 1:numel(tags)
    fprintf('-- Feature %s --\n', tags{i});
    try
        feature = node.feature(tags{i});
        try_print('Label', @() char(feature.label));
        try_print('Type', @() char(feature.getType));
        try_disp_properties(feature);
    catch err
        fprintf('Unavailable: %s\n', err.message);
    end
end
end

function try_disp_properties(node)
try
    props = mphgetproperties(node);
    if ~isempty(fieldnames(props))
        disp(props);
    end
catch
end
end

function print_mesh_stats(model)
fprintf('== Mesh statistics ==\n');
try
    comp_tags = get_tags(@() model.component().tags);
    for c = 1:numel(comp_tags)
        comp = model.component(comp_tags{c});
        mesh_tags = get_tags(@() comp.mesh().tags);
        for m = 1:numel(mesh_tags)
            fprintf('-- Component %s mesh %s --\n', comp_tags{c}, mesh_tags{m});
            info = mphmeshstats(model, mesh_tags{m});
            disp(info);
        end
    end
    if isempty(comp_tags)
        fprintf('(no component tags available)\n');
    end
catch err
    fprintf('Mesh statistics unavailable: %s\n', err.message);
end
fprintf('\n');
end

function print_solution_info(model)
fprintf('== Solution information ==\n');
try
    info = mphsolinfo(model);
    disp(info);
catch err
    fprintf('Solution information unavailable: %s\n', err.message);
end
fprintf('\n');
end

function print_tag_collection(title_text, fn)
fprintf('== %s ==\n', title_text);
tags = get_tags(fn);
print_tags(tags);
fprintf('\n');
end

function tags = get_tags(fn)
try
    raw = fn();
    if iscell(raw)
        tags = cellfun(@char, raw, 'UniformOutput', false);
    elseif isstring(raw) || ischar(raw)
        tags = cellstr(raw);
    else
        tags = cell(1, numel(raw));
        for i = 1:numel(raw)
            tags{i} = char(raw(i));
        end
    end
catch
    tags = {};
end
end

function print_tags(tags)
if isempty(tags)
    fprintf('(none or unavailable)\n');
else
    for i = 1:numel(tags)
        fprintf('- %s\n', tags{i});
    end
end
end

function try_print(name, fn)
try
    fprintf('%s: %s\n', name, fn());
catch
end
end

function try_print_inline(name, fn)
try
    fprintf('%s=%s, ', name, fn());
catch
end
end
