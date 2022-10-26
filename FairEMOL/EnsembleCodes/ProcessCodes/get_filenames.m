function files_names = get_filenames(pathname)

    files = dir(fullfile(pathname));
    files(1:2) = [];
    files_names = {};
    for i = 1:length(files)
        files_names{i} = [pathname, '/', files(i).name '/'];
    end
    
end