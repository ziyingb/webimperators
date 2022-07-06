from website import create_app
import re
app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
