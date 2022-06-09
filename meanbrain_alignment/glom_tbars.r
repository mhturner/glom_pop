# Get TBars, and which cell owns them, for each glomerulus mask
# mhturner@stanford.edu

library(nat.flybrains)
library(nat.jrcbrains)
library(nat.h5reg)
library(neuprintr)
library(dplyr)
library(bioimagetools)

options(warn=1)

data_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/sync/template_brain'
save_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/sync/template_brain/tbars_in_gloms'

t0 = Sys.time()

# Load glom map
res = 0.38 # um/voxel of atlas
glom_map <- bioimagetools::readTIF(file.path(data_dir, 'glom_mask_4_r.tif'), as.is=TRUE)



cypher = sprintf("MATCH (n :`hemibrain_Element`) WHERE distance(point({x:100, y:200, z:300}), n.location) < 100  AND n.type = 'pre'  return ID(n), n.type, n",
                   id2json(bodyids),
                   all_segments.json)
# Init results matrices
branson_count_matrix <- matrix(0, max(glom_map), max(branson_atlas))
branson_tbar_matrix <- matrix(0, max(branson_atlas), max(branson_atlas))
branson_weighted_tbar_matrix <- matrix(0, max(branson_atlas), max(branson_atlas))

ito_count_matrix <- matrix(0, max(ito_atlas), max(ito_atlas))
ito_tbar_matrix <- matrix(0, max(ito_atlas), max(ito_atlas))
ito_weighted_tbar_matrix <- matrix(0, max(ito_atlas), max(ito_atlas))

syn_mask <- array(0, dim=dim(ito_atlas))

# get synapses in PVLP/PLP
syn_data = neuprint_get_synapses((), roi=c('PVLP(L)', 'PLP(L)'), chunk=TRUE, progress=TRUE, remove.autapses=TRUE)
for (g_ind in 1:max(glom_map)){






} # end g_ind

for (c_ind in 1:length(chunks)){
  body_ids = neuprint_ids(chunks[[c_ind]])

  # get synapses associated with bodies
  syn_data = neuprint_get_synapses(body_ids)
  syn_data = syn_data[!duplicated(syn_data[2:5]),] # remove duplicate T-bar locations (single t-bar -> multiple postsynapses)

  print(sprintf('Loaded chunk %s: syn_data size = %s x %s', c_ind, dim(syn_data)[1],  dim(syn_data)[2]))

  # convert hemibrain raw locations to microns
  syn_data[,c("x", "y", "z")] = syn_data[,c("x", "y", "z")] * 8/1000 # vox -> um

  # Go from hemibrain space to comparison space
  if (comparison_space == 'JFRC2'){
    syn_data[,c("x", "y", "z")] = xform_brain(syn_data[,c("x", "y", "z")], sample = JRCFIB2018F, reference = JFRC2) / res # x,y,z um -> atlas voxels
  } else if (comparison_space == 'JRC2018') {
    syn_data[,c("x", "y", "z")] = xform_brain(syn_data[,c("x", "y", "z")], sample = JRCFIB2018F, reference = JRC2018F) / res # x,y,z um -> atlas voxels
  }

  # split into input / output
  input_synapses = as.data.frame(syn_data[syn_data$prepost==1, c("x", "y", "z", "bodyid")])
  output_synapses = as.data.frame(syn_data[syn_data$prepost==0, c("x", "y", "z", "bodyid")])

  # For each cell in synapse list
  ct = 0
  for (body_id in body_ids){
    # Swap x and y for indexing
    input_yxz = data.matrix(input_synapses[input_synapses$bodyid==body_id, c("y", "x", "z")])
    output_yxz = data.matrix(output_synapses[output_synapses$bodyid==body_id, c("y", "x", "z")])

    mode(input_yxz) = 'integer' # floor to int to index
    mode(output_yxz) = 'integer' # floor to int to index

    # # # # # # BRANSON ATLAS # # # # # # # # # # # #
    # Get regions + counts for cell inputs
    input_regions = branson_atlas[input_yxz] # index atlas matrix with integer array
    input_regions = input_regions[input_regions!=0] # remove 0 regions (non-brain)
    input_tab = table(input_regions) # https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/table
    input_regions = as.numeric(names(input_tab)) # now unique regions
    input_counts = as.vector(input_tab)
    # Same for cell outputs
    output_regions = branson_atlas[output_yxz]
    output_regions = output_regions[output_regions!=0]
    output_tab = table(output_regions)
    output_regions = as.numeric(names(output_tab))
    output_counts = as.vector(output_tab)

    if (length(input_regions) > 0 && length(output_regions) > 0){
      # Cell count
      branson_count_matrix[input_regions, output_regions] =
        branson_count_matrix[input_regions, output_regions] + 1

      # Total T-bar count
      branson_tbar_matrix[input_regions, output_regions] =
        branson_tbar_matrix[input_regions, output_regions] +
        t(replicate(length(input_regions), output_counts))

      # Weighted T-bar count: output tbars mult. by fraction of total input synapses in source region
      branson_weighted_tbar_matrix[input_regions, output_regions] =
        branson_weighted_tbar_matrix[input_regions, output_regions] +
        as.matrix(input_counts / sum(input_counts)) %*% t(as.matrix(output_counts))
    }

    # # # # # # ITO ATLAS # # # # # # # # # # # #
    # Get regions + counts for cell inputs
    input_regions = ito_atlas[input_yxz] # index atlas matrix with integer array
    input_regions = input_regions[input_regions!=0] # remove 0 regions (non-brain)
    input_tab = table(input_regions) # https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/table
    input_regions = as.numeric(names(input_tab)) # now unique regions
    input_counts = as.vector(input_tab)
    # Same for cell outputs
    output_regions = ito_atlas[output_yxz]
    output_regions = output_regions[output_regions!=0]
    output_tab = table(output_regions)
    output_regions = as.numeric(names(output_tab))  # now unique regions
    output_counts = as.vector(output_tab)

    if (length(input_regions) > 0 && length(output_regions) > 0){
      # Cell count
      ito_count_matrix[input_regions, output_regions] =
        ito_count_matrix[input_regions, output_regions] + 1

      # Total T-bar count
      ito_tbar_matrix[input_regions, output_regions] =
        ito_tbar_matrix[input_regions, output_regions] +
        t(replicate(length(input_regions), output_counts))

      # Weighted T-bar count: output tbars mult. by fraction of total input synapses in source region
      ito_weighted_tbar_matrix[input_regions, output_regions] =
        ito_weighted_tbar_matrix[input_regions, output_regions] +
        as.matrix(input_counts / sum(input_counts)) %*% t(as.matrix(output_counts))
    }

    if (length(output_yxz) > 0){
      # Append output synapse counts to synapse mask
      ct_by_vox = aggregate(data.frame(output_yxz)$x, by=data.frame(output_yxz), length)
      syn_mask[data.matrix(ct_by_vox)[,1:3]] = syn_mask[data.matrix(ct_by_vox)[,1:3]] + data.matrix(ct_by_vox)[,4]
    }
    ct = ct + 1
  } # end body_ids

  print(sprintf('Completed chunk %s: total cells = %s', c_ind, ct))

} # end chunks

# Save conn matrices and syn mask
write.csv(branson_count_matrix, file.path(save_dir, 'hemi_2_atlas', 'branson_cellcount_matrix.csv'))

Sys.time() - t0
