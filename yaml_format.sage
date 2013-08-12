import random
import re
import itertools
import string
import pylab as plt
import os
# some wildcard variables for simplifying expressions
w0=SR.wild(0);w1=SR.wild(1);w2=SR.wild(2)


def yaml_add_svg_question(qs_file, instruction_string, question_graphic=False, answer_graphics=False, question_string=False, answer_strings=False):
    if question_string is False:
        question_string = yaml_generate_svg_string(question_graphic)
    if answer_strings is False:
        answer_strings = [yaml_generate_svg_string(a) for a in answer_graphics]
    yaml_add_question(instruction_string, question_string, answer_strings, qs_file)
    
def yaml_generate_svg_string(graphic):
    graphic.save("temp.svg",figsize=5)
    svg_file = open("temp.svg")
    svg_string = svg_file.read().replace("\n","")
    svg_file.close()
    os.remove("temp.svg")
    return svg_string

def yaml_make_answer_list(answer,distractors,number_of_distractors=4):
    return make_answer_list(answer,distractors, number_of_distractors)

def yaml_create_question_set(qs_name, question_format='html', answer_format='html multiple'):
    """ sets up the header for a question set yaml file"""
    qs_file = open(qs_name,'w')
    qs_file.write('---\n')
    qs_file.write('version: 2\n')
    qs_file.write('question format: ' + question_format + '\n')
    qs_file.write('answer format: ' + answer_format + '\n')
    qs_file.write('\n')
    qs_file.write('questions:\n')
    qs_file.close()

def yaml_add_question(instruction_string, question_string, answer_strings, qs_file):
    """ adds a question with given instruction string, question string, and list of answer strings to the given file """
    qs_file.write('- question:\n')
    qs_file.write('    instruction: '+instruction_string+'\n')
    qs_file.write('    question: '+question_string+'\n')
    qs_file.write('    answers:\n')
    for answer in answer_strings:
        qs_file.write('    - '+answer+'\n')

def create_html_question_set(file_handle, question_format='html', answer_format='html multiple'):
	file_handle.write('---\n')
	file_handle.write('version: 2\n')
	file_handle.write('question format: ' + question_format + '\n')
	file_handle.write('answer format: ' + answer_format + '\n')
	file_handle.write('\n')
	file_handle.write('questions:\n')
	
def create_html_text_question_set(file_handle):
	file_handle.write('---\n')
	file_handle.write('version: 2\n')
	file_handle.write('question format: html\n')
	file_handle.write('answer format: text\n')
	file_handle.write('\n')
	file_handle.write('questions:\n')

def create_html_html_multiple_question_set(file_handle):
	file_handle.write('---\n')
	file_handle.write('version: 2\n')
	file_handle.write('question format: html\n')
	file_handle.write('answer format: html multiple\n')
	file_handle.write('\n')
	file_handle.write('questions:\n')

def insert_html_text_question(instruction,question,answer,file_handle):
	file_handle.write('- question:\n')
	file_handle.write('    instruction: '+instruction+'\n')
	file_handle.write('    question: '+question+'\n')
	file_handle.write('    answers:\n')
	file_handle.write('    - '+answer+'\n')

def insert_html_html_multiple_question(instruction,question,answer_list,file_handle):
	file_handle.write('- question:\n')
	file_handle.write('    instruction: '+instruction+'\n')
	file_handle.write('    question: '+question+'\n')
	file_handle.write('    answers:\n')
	for answer in answer_list:
		file_handle.write('    - '+answer+'\n')

def graph_line_pair_question(simple_line_object1,simple_line_object2):
	draw_grid(3)
	x_list = range(-12,13)
	y_list = [simple_line_object1.slope_intercept_form().rhs().subs(x=x_value) for x_value in x_list]
	plt.plot(x_list,y_list,color="grey", linewidth=2.0, linestyle="-")
	y_list = [simple_line_object2.slope_intercept_form().rhs().subs(x=x_value) for x_value in x_list]
	plt.plot(x_list,y_list,color="grey", linewidth=2.0, linestyle="-")
		
	plt.savefig('temp.svg')
	svg_file = open('temp.svg')
	svg_string = svg_file.read()
	svg_file.close()	
	question_string = "|\n\n              " + svg_string.replace('\n','')
	return question_string

def graph_line_pair_answer(simple_line_object1,simple_line_object2):
	draw_grid(3)
	x_list = range(-12,13)
	y_list = [simple_line_object1.slope_intercept_form().rhs().subs(x=x_value) for x_value in x_list]
	plt.plot(x_list,y_list,color="grey", linewidth=2.0, linestyle="-")
	y_list = [simple_line_object2.slope_intercept_form().rhs().subs(x=x_value) for x_value in x_list]
	plt.plot(x_list,y_list,color="grey", linewidth=2.0, linestyle="-")
	plt.savefig('temp.svg')
	svg_file = open('temp.svg')
	svg_string = svg_file.read()
	svg_file.close()
	question_string = "|\n      " + svg_string.replace('\n','')
	return question_string


class simple_quadratic:
	"""multiple representations and elements of a quadratic:
	defines line in terms of a vertex and a scaling_factor; a(x-h)^2 + k
	assumes coordinates are integers between -6 and 6'
	"""
	point_1 = ()
	point_2 = ()

	def __init__(self,vertex,scaling_factor):
		self.vertex = vertex
		self.scaling_factor = scaling_factor

	def __repr__(self):
		return "parabola with vertex "+str(self.vertex)+" and scaling factor "+str(self.scaling_factor)

        def vertex(self):
	        return self.vertex
  
        def vertex_form(self):
		return self.scaling_factor*(x-self.vertex[0])^2 + self.vertex[1]

	def standard_form(self):
	        a = self.scaling_factor
		h,k = self.vertex
		b = -2*a*h
		c = a*h^2 + k
   		return a*x^2 + b*x + c

	def standard_form_a(self):
		return self.scaling_factor

	def standard_form_b(self):
		return -2*self.scaling_factor*self.vertex[0]

	def standard_form_c(self):
		return self.scaling_factor * self.vertex[0]^2 + self.vertex[1]

	def discriminant(self):
		a,b,c = self.standard_form_a(),self.standard_form_b(),self.standard_form_c()
		return b^2 - 4*a*c

	def type_of_roots(self):
		if self.discriminant() > 0:
		    type = "two real"
		elif self.discriminant() == 0:
		    type = "one real"
		elif self.discriminant() < 0:
		    type = "two complex"
		return type

	def two_roots(self):
		solutions = solve(self.vertex_form()==0,x)
		return [solution.rhs() for solution in solutions]
					
	def graph_quadratic_question(self):
		draw_grid(3)
		x_list = range(-12,13)
		y_list = [self.standard_form().subs(x=x_value) for x_value in x_list]
		plt.plot(x_list,y_list,color="grey", linewidth=2.0, linestyle="-")
		plt.savefig('temp.svg')
		svg_file = open('temp.svg')
		svg_string = svg_file.read()
		svg_file.close()	
		question_string = "|\n\n              " + svg_string.replace('\n','')
		return question_string
	def graph_quadratic_answer(self):
		draw_grid(3)
		x_list = range(-12,13)
		y_list = [self.standard_form().subs(x=x_value) for x_value in x_list]
		plt.plot(x_list,y_list,color="grey", linewidth=2.0, linestyle="-")
		plt.savefig('temp.svg')
		svg_file = open('temp.svg')
		svg_string = svg_file.read()
		svg_file.close()
		question_string = "|\n      " + svg_string.replace('\n','')
		return question_string

# create an html table of class vertical_table from a 2-d array (list of lists)
# assumes it is being passed latex strings as the contents of each cell.
# issues: I had wanted to add the \( \) delimiters around each element of the table, but
# the python code escapes the \.
# for now, I've added the double slashes back in, but it means that you'll have to convert 
# \\ to \ in the .txt file as a global find and replace for this to work.
# also, this creates a table of class="vertical_table", but we should really have the flexibility
# of a horizontal or vertical table, with header row or header column that is distinguished from the 
# other rows (or columns).

def html_table(data):
	td_data = [['<td>\('+data[row][col]+'\)</td>' for col in range(len(data[0]))] for row in range(len(data))]
	tr_data = ['<tr>'+ ''.join(row) + '</tr>' for row in td_data]
	table = '<table class="vertical_table" border="1">' + ''.join(tr_data) + '</table>'
	return table
    
def yaml_html_table(data):
    table = "<table class='vertical_table' border='1'>"
    for row in data: 
        table += "<tr>"
        for cell in row:
            table += "<td>%s</td>" % cell
        table += "</tr>"
    table += "</table>"
    return table

# transpose data[row][column] into data_prime[column][row]
def transpose_array(data):
	return [ [data[row][col] for row in range(len(data))] for col in range(len(data[0])) ]

# convert a list of 5 sage output answers into a string of 
# latex equivalent, yaml-formatted answers.
def convert_answers_from_sage_to_yaml(sage_answers):
    latex_list= [latex(object) for object in sage_answers]
    string = '    answers:'+'\n'
    string = string + '    -' + latex_list[0]+ '\n'
    string = string + '    -' + latex_list[1]+ '\n'
    string = string + '    -' + latex_list[2]+ '\n'
    string = string + '    -' + latex_list[3]+ '\n'
    string = string + '    -' + latex_list[4]+ '\n'

    return string

def reformat_fraction(fraction_expression):
# nicer formulation of fractions.  ex. instead of \\frac{2}{3} \\, \\pi we want \\frac{2 \\, \\pi}{3}
# as well as \\frac{2 \\, \\sqrt{3}}{3} instead of \\frac{2}{3} \\, \\sqrt{3}  
# the if statement is for non-fractional expressions, for example, when  the angle is 0 or pi or 2*pi
# this includes special treatment in the case when the numerator in the fraction is 1.
# improvements: is it possible to do this with the expressions, 
# using match and subs, rather than waiting for the latex?

			reformat_fraction_prep = re.search(r"(?P<sign>[-]*)\\frac{(?P<frac1>\d+)}{(?P<frac2>\d+)}\s\\,\s(?P<expr1>.*)",latex(fraction_expression))
			if reformat_fraction_prep:
				sign1 = reformat_fraction_prep.group('sign')
				fraction1 = reformat_fraction_prep.group('frac1')
				fraction2 = reformat_fraction_prep.group('frac2')
				expression1 = reformat_fraction_prep.group('expr1')

				if reformat_fraction_prep.group('frac1') == '1':	
					reformat_latex_fraction = sign1+'\\frac{'+ expression1 +'}{'+reformat_fraction_prep.group('frac2')+'}'
				else:
					reformat_latex_fraction = sign1+'\\frac{'+reformat_fraction_prep.group('frac1')+'\\,'+ expression1 +'}{'+reformat_fraction_prep.group('frac2')+'}'
				return reformat_latex_fraction
			else:
				return fraction_expression

# latex of  exp(f(x)) is unsightly. Replaces it with e^{f(x)}
# improvements: is it possible to do this with the expressions,	
# using match and	s\ubs, rather than waiting for the latex?
def reformat_exp(latex_expression):
    return re.sub(r'(?P<lhs>e\^{\\left\()(?P<input>.*?)(?P<rhs>\\right\)})',r'e^{\g<input>}',latex_expression)

# remove the answer from the distractor set, if present, and then append.
# 6-15-2012: note that there is some legacy use of this which assumes exactly 4 distractors.


def make_answer_list(answer,distractors, number_of_distractors=4):
	temp_list=distractors[:]
	for i in range(temp_list.count(answer)):
	    temp_list.remove(answer)
	unique_distractors = my_uniquify(temp_list)
	if len(unique_distractors) >= number_of_distractors:
		answer_list = random.sample(unique_distractors, number_of_distractors)
		answer_list.append(answer)
		answer_list.reverse()
		return answer_list
	else:
		return False 

def consider_alternate_distractors(answer,distractors,alternate_distractors=[],number_of_distractors=4):
	temp_list = distractors[:]
	alt_temp_list = alternate_distractors[:]
	while answer in temp_list:
	    temp_list.remove(answer)
	while answer in alt_temp_list:
	    alt_temp_list.remove(answer)
	for distractor in temp_list:
	    while distractor in alt_temp_list:
	        alt_temp_list.remove(distractor)
	unique_distractors = my_uniquify(temp_list)
	unique_alt_distractors = my_uniquify(alt_temp_list)
	if len(unique_distractors) >= number_of_distractors:
	    answer_list = random.sample(unique_distractors, number_of_distractors)
	    answer_list.append(answer)
	    answer_list.reverse()
	    return answer_list
	elif len(unique_distractors + unique_alt_distractors) >= number_of_distractors:
	    answer_list = unique_distractors
	    answer_list.extend(random.sample(unique_alt_distractors,number_of_distractors-len(answer_list)))
	    answer_list.append(answer)
	    answer_list.reverse()
	    return answer_list
	else:
	    return False



# my_uniquify is a helper function for make_answer_list.
# it is not used in the current version but may need to be used in the future in place
# of this line of code:
# unique_distractors = list(set(a_list)) , which removes duplicates from the list of distractors.
# set requires the ability to hash, and when our list elements start to include equations,
# we can no longer use that.
# instead we need to use unique_distractors = my_uniquify(temp_list)
def my_uniquify(list):
    output = []
    for element in list:
    	if element not in output:
	   output.append(element)
    return output	    


def tan_squared_trig_identity(expression):
    return expression.subs(tan(w0)^2+1==sec(w0)^2)

# number line class definition

class number_line(object):
	def __init__(self, radius=10):

		self.radius = radius
		self.span = radius * 1.1 #edges of the graph. leaves room for arrows
		self.ceil = radius + 1 #if a plot goes to this point it means it is going to infinity
		self.floor = -radius - 1 #if a plot goes to this point it means it is going to negative infinity
		
		self.black_linewidth = 3
		self.red_linewidth = 5
		self.arrow_width = 3
		
		self.circle_size = 11.5
		self.circle_edge_width = 1.5
		
		self.setup()
		self.drawLine()
		
	def setup(self):
		self.fig1 = plt.figure( facecolor='white') #whole background is white
		self.ax = plt.axes(frameon=False) #frame is hidden
		self.ax.axes.get_yaxis().set_visible(False) # turn off y axis
		self.ax.get_xaxis().tick_bottom() #disable ticks at top of plot
		self.ax.axes.get_xaxis().set_ticks([]) #sets ticks on bottom to blank
		plt.axis([-self.span, self.span, -1, 1]) # sets dimensions of axes to xmin, xmax, ymin, ymax
	
	def drawLine(self):
		#plot the main line  (slightly crooked so svg will render)
		plt.plot([-self.radius,self.radius],[0,0.001],'k',linewidth=self.black_linewidth,zorder=1) 


		#tick marks and labels
		for i in range(-self.radius,self.radius+1):
			length = .03
			tick_y = -.1
			if i==0: 
				length=.05
				tick_y = -.13
			 #plot the tick marks (slightly crooked so svg will render)
			plt.plot([i,i-.001],[-length,length],'k',linewidth=2, zorder=1) 
			
			#write the labels
			plt.text(i,tick_y,str(i),horizontalalignment='center')
		
		#draw the arrows
		self.arrow((self.span,0),'k',facingRight = True)
		self.arrow((-self.span,0),'k',facingRight = False)


	def arrow(self, coord, color,facingRight=True):
		tilt = 20 #horizontal left
		if  facingRight: tilt = -20 #horizontal right
		plt.annotate('', xy=coord,  xycoords='data',
			xytext=(tilt, 0), textcoords='offset points',
			arrowprops=dict(arrowstyle="->", linewidth=self.arrow_width , mutation_scale=18,color=color))
	
	#graph a series of inequalities
	def graph(self,eqs):
		

		
		toGraph = []
		for eq in eqs:
			toGraph = self.solve(eq,toGraph)

		for piece in toGraph:
			
			#this piece of the graph is a point
			if len(piece)==1: 
				plt.plot(piece,[0],'or', markersize = self.circle_size, markeredgewidth = self.circle_edge_width, zorder=10)
			
			# this piece of the graph is a line
			elif len(piece)==2: 
			
				#line going to negative infinity with arrow
				if piece[0]==self.floor: 
				
					plt.plot([piece[1]],[0],'or',markerfacecolor='white',markeredgecolor='r',markersize = self.circle_size, markeredgewidth = self.circle_edge_width, zorder=10)
					plt.plot([-self.radius,piece[1]],[0,0],'-r',linewidth=self.red_linewidth,zorder=5)
					self.arrow((-self.span,0),'r',facingRight = False)
				
				#line going to positive infinity with arrow
				if piece[1]==self.ceil:
				
					plt.plot([piece[0]],[0],'or',markerfacecolor='white',markeredgecolor='r',markersize = self.circle_size, markeredgewidth = self.circle_edge_width,zorder=10)
					plt.plot([piece[0],self.radius],[0,0],'-r',linewidth=self.red_linewidth,zorder=5)
					self.arrow((self.span,0),'r',facingRight = True)
				
	#helper method for graph turns the equations into coordinates
	def solve(self,eq, toGraph):
	    
		
	    if '>' in eq:
	        toGraph.append([self.getAfter(eq,'>') , self.ceil])
	    elif '<' in eq:
	        toGraph.append([self.floor,self.getAfter(eq,'<')])
	
	    if '=' in eq:
	        toGraph.append([self.getAfter(eq,'=')])
	    return toGraph
	
	#helper method for solve helps parse the equations
	def getAfter(self,s, char ):
		if'>=' in s: char = '>='
		if '<=' in s: char = '<='
		return float(string.split(s,char)[-1])
		
	#clears anything that is graphed
	def clear(self):
		plt.close()
		self.setup()
		self.drawLine()
	

#helper functions for working with inequalities
def opposite(sign):
	if sign == ">": return "<"
	if sign == "<": return ">"
	if sign == ">=": return "<="
	if sign == "<=": return ">="
	if sign == "\geq": return "\leq"
	if sign == "\leq": return "\geq"
	if sign == gt: return lt
	if sign == lt: return gt
	if sign == le: return ge
	if sign == ge: return le
	
def wrong_circle(sign):
	if sign == "\leq": return "<"
	if sign == "\geq": return ">"
	if sign == gt: return ge
	if sign == ge: return gt
	if sign == lt: return le
	if sign == le: return lt
	if "=" in sign: return sign[0]
	else: return sign + "="
		
def latex_sign(sign):
	if (sign == "<=") or (sign == le): return "\leq"
	if (sign == ">=") or (sign == ge): return "\geq"
	if sign == lt: return "<"
	if sign == gt: return ">"
	else: return sign
	
'''
This is dmitry's contribution of the basic cartesian grid (draw_grid).

'''
#import matplotlib
#matplotlib.use('SVG')
from matplotlib.colors import NP_MAJOR

def setup_grid_font():
    # setup font
    import matplotlib.pylab as plt
    from matplotlib import rc
    #plt.rcParams['ps.useafm'] = True
    rc('font',**{'family':'serif','serif':['Arial', 'Helvetica', 'Verdana', 'Geneva', 'Lucid', 'Arial', 'Avant Garde', 'Bitstream Vera Sans', 'sans-serif']})
    plt.rcParams['pdf.fonttype'] = 42


MIN_X = -12
MAX_X = 12
MIN_Y = -12
MAX_Y = 12
major_grid_line_width = 0.25 #0.75
minor_grid_line_width = 0.25

def make_labels(min_, max_, mod_):
    for i in range(min_, max_):
        if i == 0:
            yield ' '
        elif i % mod_ == 0:
            yield str(i)

def get_origins(lines):
    lines_count = len(lines)
    return [lines_count / 4, lines_count / 4 - 1, 3 * lines_count / 4, 3 * lines_count / 4 - 1]


def draw_grid(overall_size=3):
    ##Uncomment the following line and change setup_grid_font() 
    ##in order to use some specific fonts
    #setup_grid_font()
    from matplotlib.pylab import figure, axes, MultipleLocator
    ##Note: saving to svg file and plotting would require following imports
    #from matplotlib.pylab import np, plot, savefig, show

    # Create a new figure of size 1:1 360x360
    fig = figure(figsize=(overall_size, overall_size), dpi=72)
    fig.set_facecolor('white')
    fig.set_edgecolor('white')

    # Set inner view port for grid and left some space at top and right for x,y labels
    ax = axes([0.02,0.02,0.92,0.92])

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))

    # set limits and ticks
    ax.set_xlim(MIN_X, MAX_X)
    ax.set_ylim(MIN_Y, MAX_Y)

#    ax.set_xticks(np.linspace(MIN_X + 2, MAX_X - 2, 2 * MAX_X + 1, endpoint=True))
#    ax.set_yticks(np.linspace(MIN_Y + 2, MAX_Y - 2, 2 * MAX_Y + 1, endpoint=True))

    ax.xaxis.set_major_locator(MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.yaxis.set_minor_locator(MultipleLocator(1.0))

    #Set ticks size
    #Major ticks
    lines = ax.get_xticklines() + ax.get_yticklines()
    skip = get_origins(lines)
    for l in lines:
        if lines.index(l) in skip: #[4,5, 14,15]:
            #skip origin
            continue

        l.set_markersize(6)
        l.set_markeredgewidth(2)

    #Minor ticks
    lines = ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines(minor=True)
    skip = get_origins(lines)
    for l in lines:
        if lines.index(l) in skip:
            l.set_markersize(0)
            continue
        l.set_markersize(4)
        l.set_markeredgewidth(2)

    ax.grid(which='major', axis='x', linewidth=major_grid_line_width, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='x', linewidth=minor_grid_line_width, linestyle='-', color='0.75')
    ax.grid(which='major', axis='y', linewidth=major_grid_line_width, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='y', linewidth=minor_grid_line_width, linestyle='-', color='0.75')

    X_labels = list(make_labels(MIN_X, MAX_X, 5))
    Y_labels = list(make_labels(MIN_Y, MAX_Y, 5))

    ax.set_xticklabels(X_labels)
    ax.set_yticklabels(Y_labels)

    #Set the semi-transparent background for tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(14)
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 ))

    #Draw 'x' 'y' labels
    ax.text(MAX_X + 0.5, 0, r'$x$', ha='left', size=16)
    ax.text(0, MAX_Y + 0.5, r'$y$', ha='left', size=16)


def create_set_of_linear_equations(point_range=1):
	'''create lines through all the (non-identical) point pairs of the given lattice 
		eliminates duplicate lines (i.e., different point pairs, but same representation in standard form)
	'''
	points = [(i,j) for i in range(-point_range,point_range+1 ) for j in range(-point_range,point_range+1 )]
	base_equation_set = [simple_line(point_1,point_2) for point_1 in points for point_2 in points if (point_1 != point_2 and random.random()<.06)]
	shadow_unique_set = []
	unique_equation_set = []
	for equation in base_equation_set:
		if equation.standard_form() not in shadow_unique_set:
			shadow_unique_set.append(equation.standard_form())
			unique_equation_set.append(equation)
	return unique_equation_set





class simple_line:
	"""multiple representations of a line:
	defines line in terms of two points
	assumes coordinates are integers between -6 and 6'
	"""
	point_1 = ()
	point_2 = ()

	def __init__(self,point_1,point_2):
		self.point_1 = point_1
		self.point_2 = point_2

	def __repr__(self):
		return "line thru " + str(self.point_1)+ " and " + str(self.point_2)

	def slope(self):
		try:
			return Integer(self.point_2[1]-self.point_1[1])/Integer(self.point_2[0]-self.point_1[0])
		except ZeroDivisionError:
			return 'undefined'	
	
	def type_of_slope(self):
		if self.slope() == 'undefined':
			return 'undefined'
		elif self.slope() == 0:
			return 'zero'
		elif self.slope() > 0:
			return 'positive'
		else:
			return 'negative'			

	def y_intercept(self):
		if self.slope() == 'undefined':
			return 'undefined'
		else:
			return self.point_1[1] - self.slope() * self.point_1[0]
		
	def x_intercept(self):
		if self.slope() == 0:
			return 'undefined'
		if self.y_intercept() == 'undefined':
			return self.point_1[0] 
		else:	
			return -1 * self.y_intercept() / self.slope() 
			
	def slope_intercept_form(self):
		if self.slope() == 'undefined':
			return 'undefined'
		else:
			x = var('x')
			y = var('y')
			return y == self.slope() * x + self.y_intercept()
		
	def standard_form(self):
		if self.slope() == 'undefined':
			return x == self.point_1[0]
		else:
			temp = self.slope_intercept_form()-self.slope()*x
			return 	temp*self.slope().denominator()
			
	def standard_form_x_coefficient(self):
		lhs = self.standard_form().lhs()
		try:
			return lhs.coefficients(x)[1][0]
		except IndexError:
			if 	lhs.coefficients(x)[0][1]==1:
				return lhs.coefficients(x)[0][0]
			else:
				return 0	

	def standard_form_y_coefficient(self):
		lhs = self.standard_form().lhs()
		try:
			return lhs.coefficients(y)[1][0]
		except IndexError:
			if 	lhs.coefficients(y)[0][1]==1:
				return lhs.coefficients(y)[0][0]
			else:
				return 0	

	def standard_form_constant_coefficient(self):
		return self.standard_form().rhs()
		
	def graph_line_question(self):
		draw_grid(3)
		x_list = range(-12,13)
		y_list = [self.slope_intercept_form().rhs().subs(x=x_value) for x_value in x_list]
		plt.plot(x_list,y_list,color="grey", linewidth=2.0, linestyle="-")
		plt.savefig('temp.svg')
		svg_file = open('temp.svg')
		svg_string = svg_file.read()
		svg_file.close()	
		question_string = "|\n\n              " + svg_string.replace('\n','')
		return question_string
		
	def graph_points(self):
		draw_grid(3)
		x_list = [self.point_1[0], self.point_2[0]]
		y_list = [self.point_1[1], self.point_2[1]]
		plt.plot(x_list,y_list, 'ro')
		plt.savefig('temp.svg')
		svg_file = open('temp.svg')
		svg_string = svg_file.read()
		svg_file.close()	
		question_string = "|\n\n              " + svg_string.replace('\n','')
		return question_string
		
	def graph_line_answer(self):
		draw_grid(3)
		x_list = range(-12,13)
		y_list = [self.slope_intercept_form().rhs().subs(x=x_value) for x_value in x_list]
		plt.plot(x_list,y_list,color="grey", linewidth=2.0, linestyle="-")
		plt.savefig('temp.svg')
		svg_file = open('temp.svg')
		svg_string = svg_file.read()
		svg_file.close()
		question_string = "|\n      " + svg_string.replace('\n','')
		return question_string