from IPython.display import HTML
from IPython.display import display

def add():

    tag = HTML('''
    
    <script>
    //$( document ).ready(function() {
        
    var add_button = function () {
        Jupyter.notebook.get_cells().forEach(function(cell) {
            if (cell.element.find("form.bla").length == 0) {
                cell.element.find("div.output_wrapper").append('<form class="bla" action="javascript:toggle_selected_input()"><input type="submit" style="float: right;" value="T"></i></form>');
            }
        })
    };
    
    var toggle_selected_input = function () {
        // Find the selected cell
        var cell = Jupyter.notebook.get_selected_cell();
        // Toggle visibility of the input div
        cell.element.find("div.input").toggle('slow');
        cell.metadata.hide_input = ! cell.metadata.hide_input;
        };
    
    
    add_button()    

      
    //} );
    </script>
    


    ''')
    
    return display(tag)