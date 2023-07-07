# ML_SC_Project # 
Computational methods for predicting materials properties has a long and very successful history, but some phenomena have not been completely described by analytical or computational methods. One such phenomenon of particular interest is superconductivity. How materials properties interact to give rise to superconductivity and how they contribute to the critial temperature are of particular interest. Because of the wide range of materials properties that interect in subtle, often surprising ways, answering these questions and predicting critial temperatures is a very natural problem to apply machine learning methods to. Several attempts have been made<sup>[1][2]</sup>, but they are limited in their representation of materials
## Representing Materials ##  
Several attempts have been made to represent materials<sup>[3][4]</sup>, but most sacrifice features for scalability. One of the fundamential issues with representing materials for computational analysis is the lack of fixed length input; some materials simply have more constituents, or a more complicated stucture. This makes representing them for use in machine learning algorithms particularly difficult. Compounding this issue, there are only on the order of 30,000 different superconductors; given large enough training data, a machine learning algorithm should be able to sort through important and unimportant features, but with such a limited selection of training data, this is not possible. These issues make the represnetation of materials in this task particularly important.   

There are several things that are obvioulsy of importance, such as the compositon and geometry of the material, but represneting this data in the most effective way for an algorithm to use is not clear. The material can be broken into essentially two parts listed above, combined in a tensor as input data. The tensor will be of a fixed size, with up to 20 crystal sites being accomadated.
<h4>Representing Composition</h4>
Compostion can be represented as tensors of crystallographic sites. Each site with be a vector with element, location, and coordination enviorment. Element will simply be represented using atomic number, location using the x, y, z coordinates in lattice coordinates and the coordination environment will be an integer coorsponding to the environment. These elements will then be added to a fixed width tensor. Zero padding will be used for materials that have less than 20 crystal sites. For example, CoO would be represented as

<centre>[[27,1/3,2/3,0.417688,8],[8,1/3,2/3,0.665645],[0,0,0,0,0],...,[0,0,0,0,0]]</centre>

<h6>List of coordination environment tags</h6>

<h4>Representing Geometry</h4>
There are many ways to represnet geometry, 

## References ##
[1]Pogue, Elizabeth & New, Alexander & McElroy, Kyle & Le, Nam & Pekala, Michael & McCue, Ian & Gienger, Eddie & Domenico, Janna & Hedrick, Elizabeth & McQueen, Tyrel & Wilfong, Brandon & Piatko, Christine & Ratto, Christopher & Lennon, Andrew & Chung, Christine & Montalbano, Timothy & Bassen, Gregory & Stiles, Christopher. (2022). Closed-loop machine learning for discovery of novel superconductors. 10.48550/arXiv.2212.11855. 

[2]Quinn Margaret R., McQueen Tyrel M. (2022) Identifying New Classes of High Temperature Superconductors With Convolutional Neural Networks, Frontiers in Electronic Materials https://www.frontiersin.org/articles/10.3389/femat.2022.893797   

[3]Goodall, R.E.A., Lee, A.A. Predicting materials properties without crystal structure: deep representation learning from stoichiometry. Nat Commun 11, 6280 (2020). https://doi.org/10.1038/s41467-020-19964-7

[4]Cheng, J., Zhang, C. & Dong, L. A geometric-information-enhanced crystal graph network for predicting properties of materials. Commun Mater 2, 92 (2021). https://doi.org/10.1038/s43246-021-00194-3

