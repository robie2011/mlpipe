# import unittest
# import os
# from pathlib import Path
# from textx import metamodel_from_file, get_children_of_type, TextXSyntaxError
#
# grammer_file = os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent, "dsl", "grammer.tx")
#
#
# def load_grammer_and_parse_model(model_str: str):
#     meta = metamodel_from_file(grammer_file)
#     return meta.model_from_str(model_str)
#
#
# class MyTestCase(unittest.TestCase):
#     def test_detect_chain(self):
#         model = load_grammer_and_parse_model("chain {}")
#         self.assertIsNotNone(model.chain)
#
#     def test_chain_simple_structures(self):
#         model_str = """
#         chain {
#             process {}
#             process() {}
#             aggregate(window:'30m'){}
#             aggregate(window: "30m"){}
#         }
#         """
#         model = load_grammer_and_parse_model(model_str)
#         self.assertIsNotNone(model.chain)
#         self.assertIsNotNone(model.chain.pipes)
#
#         self.assertEqual(model.chain.pipes[2].annotation.properties[0].key, "window")
#         self.assertEqual(model.chain.pipes[2].annotation.properties[0].value, "30m")
#
#         self.assertEqual(model.chain.pipes[3].annotation.properties[0].key, "window")
#         self.assertEqual(model.chain.pipes[3].annotation.properties[0].value, "30m")
#
#     def test_chain_one_pipe_name(self):
#         model_str = """
#         chain {
#             process {
#                 processor.ColumnDropper
#             }
#         }
#         """
#         model = load_grammer_and_parse_model(model_str)
#         self.assertEqual(model.chain.pipes[0].elements[0].name, "processor.ColumnDropper")
#
#     def test_chain_one_pipe_list_values_integer(self):
#         model_str = """
#         chain {
#             process {
#                 processor.ColumnDropper: 13, 23, 2321
#             }
#         }
#         """
#         model = load_grammer_and_parse_model(model_str)
#         self.assertEqual(model.chain.pipes[0].elements[0].name, "processor.ColumnDropper")
#
#     def test_chain_one_pipe_list_values_integer(self):
#         model_str = """
#         chain {
#             process {
#                 processor.ColumnDropper: "Hello", "World"
#             }
#         }
#         """
#         model = load_grammer_and_parse_model(model_str)
#         self.assertEqual(model.chain.pipes[0].elements[0].name, "processor.ColumnDropper")
#
#     def test_chain_one_pipe_list_values_invalid_value(self):
#         model_str = """
#         chain {
#             process {
#                 processor.ColumnDropper: Hello, World
#             }
#         }
#         """
#         self.assertRaises(
#             TextXSyntaxError,
#             lambda: load_grammer_and_parse_model(model_str)
#         )
#
#     def test_chain_one_pipe_key_values(self):
#         model_str = """
#         chain {
#             process {
#                 processor.OutlierRemover {
#                      name: 23
#                 }
#             }
#         }        """
#         model = load_grammer_and_parse_model(model_str)
#         # self.assertEqual(model.chain.pipes[0].elements[0].name, "processor.OutlierRemover")
#         #
#         # self.assertEqual(model.chain.pipes[0].elements[0].config.keyvalues[0].key, "max")
#         # self.assertEqual(model.chain.pipes[0].elements[0].config.keyvalues[0].value, 23)
#         #
#         # self.assertEqual(model.chain.pipes[0].elements[0].config.keyvalues[1].key, "min")
#         # self.assertEqual(model.chain.pipes[0].elements[0].config.keyvalues[1].value, -10)
#
#     def test_chain_one_pipe_key_values(self):
#         model_str = """
#         chain {
#             process {
#                 processor.OutlierRemover [
#                     name: temp1, max: 23, min: -10
#                     name: temp2, max: 30, min: -100
#                 ]
#             }
#         }
#         """
#         model = load_grammer_and_parse_model(model_str)
#         self.assertEqual(model.chain.pipes[0].elements[0].name, "processor.OutlierRemover")
#
#         self.assertEqual(model.chain.pipes[0].elements[0].config.keyvalues[0].key, "max")
#         self.assertEqual(model.chain.pipes[0].elements[0].config.keyvalues[0].value, 23)
#
#         self.assertEqual(model.chain.pipes[0].elements[0].config.keyvalues[1].key, "min")
#         self.assertEqual(model.chain.pipes[0].elements[0].config.keyvalues[1].value, -10)
#
#
# if __name__ == '__main__':
#     unittest.main()
